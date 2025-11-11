"""
Predict the genre of a single WAV file using a trained model.

Example:
  PYTHONPATH=. python predict.py \
    --data_root /path/to/GTZAN/genres_original \
    --checkpoint artifacts/best_model.pt \
    --model resnet18 \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 --duration 7.5 \
    --wav /path/to/file.wav --topk 5
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from transforms.audio import get_mel_transform, wav_to_logmel
from utils.device import get_device, get_device_name
from utils.models import build_model

# Canonical GTZAN order, used as a last-resort fallback
GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


# ------------------------------
# Utilities
# ------------------------------
def _read_json_classmap(p: Path):
    """Return list of class names from a JSON file that is either:
       - {"idx2name": [...]} or
       - ["classA", "classB", ...]
    """
    with open(p, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "idx2name" in data:
        names = data["idx2name"]
    elif isinstance(data, list):
        names = data
    else:
        raise ValueError(f"Unsupported class_map format in {p}")
    if not isinstance(names, list) or not all(isinstance(s, str) for s in names):
        raise ValueError(f"Invalid class_map content in {p}")
    return names


def _infer_classes_from_data_root(data_root: Path):
    """Infer class names from folder names under data_root or data_root/genres."""
    root = data_root / "genres" if (data_root / "genres").exists() else data_root
    if not root.exists():
        return []
    names = [d.name for d in root.iterdir() if d.is_dir()]
    present = [g for g in GTZAN_GENRES if g in names]
    return present if len(present) >= 2 else sorted(names)


def _detect_num_classes_from_state_dict(state_dict: dict) -> int:
    """Heuristic: read classifier head out_features from checkpoint weights."""
    candidates = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim == 2:  # linear weight
            out_features, in_features = v.shape
            if re.search(r"(fc|classifier|head).*\.weight$", k):
                candidates.append((k, out_features, True))
            else:
                candidates.append((k, out_features, False))
    for k, out_features, is_head in candidates:
        if is_head:
            return out_features
    if candidates:
        k, out_features, _ = max(candidates, key=lambda t: t[1])
        return out_features
    raise RuntimeError("Could not detect num_classes from checkpoint state_dict.")


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"], state
    return state, None


def load_audio(path: Path, target_sr: int, duration: float) -> torch.Tensor:
    """Load and resample mono audio to fixed length [1, T]. Center-crop/pad.
       NOTE: returns a **CPU** tensor to keep STFT/mel on CPU (avoids MPS window mismatch).
    """
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != target_sr:
        x = resample_poly(x, target_sr, sr).astype(np.float32)
    N = int(duration * target_sr)
    if len(x) < N:
        pad = N - len(x)
        left = pad // 2
        right = pad - left
        x = np.pad(x, (left, right))
    else:
        start = max(0, (len(x) - N) // 2)
        x = x[start:start + N]
    return torch.from_numpy(x).unsqueeze(0)  # CPU [1, T]


# ------------------------------
# Inference
# ------------------------------
def predict_one(
    wav_path: Path,
    model: torch.nn.Module,
    mel_t,                      # keep mel transform on CPU
    device,
    class_names,
    sr,
    topk=5,
    out_dir=None,
    duration: float = 7.5,
):
    """Compute log-mel spectrogram (on CPU), then move to device for inference."""
    # --- keep waveform on CPU to avoid STFT window device mismatch on MPS ---
    wav = load_audio(wav_path, target_sr=sr, duration=duration)            # CPU [1, T]
    logmel = wav_to_logmel(wav, sr=sr, mel_transform=mel_t)                # CPU [1, n_mels, time]
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)              # CPU normalize

    # Now move features to device and run the model
    model.eval()
    with torch.no_grad():
        feats = logmel.unsqueeze(0).to(device)  # [B=1, 1, n_mels, time]
        logits = model(feats)
        probs = torch.softmax(logits, dim=1)[0]
        topv, topi = probs.topk(topk)
        topv, topi = topv.cpu().numpy(), topi.cpu().numpy()

    print(f"\n🎧 File: {wav_path.name}")
    print("Top predictions:")
    for rank, (p, i) in enumerate(zip(topv, topi), start=1):
        label = class_names[i] if 0 <= i < len(class_names) else f"class_{i}"
        print(f"  {rank}. {label:<12} {p*100:5.2f}%")

    # Visualization (still using the CPU tensor)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(logmel.squeeze(0).cpu(), origin="lower", aspect="auto")
    ax.set_title(f"Predicted: {class_names[topi[0]] if topi[0] < len(class_names) else f'class_{topi[0]}'}")
    plt.xlabel("Time frames")
    plt.ylabel("Mel bins")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"{wav_path.stem}_pred.png"
        plt.savefig(out_file)
        print(f"Saved spectrogram to {out_file}")
    else:
        plt.show()
    plt.close(fig)


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict genre of a single WAV file")
    ap.add_argument("--wav", type=str, required=True, help="Path to WAV file")
    ap.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt")
    ap.add_argument("--model", type=str, default="resnet18", choices=["smallcnn", "resnet18"])
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--duration", type=float, default=7.5)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="pred_artifacts")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {get_device_name()} ({device})")

    # 1) Load checkpoint first, detect its num_classes
    ckpt_path = Path(args.checkpoint)
    state_dict, raw_bundle = _load_checkpoint(ckpt_path, device)
    ckpt_num_classes = _detect_num_classes_from_state_dict(state_dict)

    # 2) Load class names (prefer artifacts/class_map.json)
    class_names = None
    art_json = Path("artifacts") / "class_map.json"
    if art_json.exists():
        try:
            class_names = _read_json_classmap(art_json)
        except Exception as e:
            print(f"Warning: could not parse {art_json}: {e}")
    if not class_names:
        dr_json = Path(args.data_root) / "class_map.json"
        if dr_json.exists():
            try:
                class_names = _read_json_classmap(dr_json)
            except Exception as e:
                print(f"Warning: could not parse {dr_json}: {e}")

    if not class_names:
        inferred = _infer_classes_from_data_root(Path(args.data_root))
        class_names = inferred if len(inferred) >= 2 else GTZAN_GENRES
        print(f"Loaded classes from data_root inference: {class_names}")
    else:
        print(f"Loaded {len(class_names)} classes from class_map.json")

    # 3) If class-name count mismatches checkpoint head, adapt to checkpoint
    if len(class_names) != ckpt_num_classes:
        print(
            f"Note: class count mismatch (class_map={len(class_names)} vs ckpt={ckpt_num_classes}). "
            f"Adapting to checkpoint."
        )
        if ckpt_num_classes == len(GTZAN_GENRES):
            class_names = GTZAN_GENRES
        else:
            class_names = [f"class_{i}" for i in range(ckpt_num_classes)]

    # 4) Build model with the checkpoint's class count
    model = build_model(args.model, num_classes=len(class_names)).to(device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {ckpt_path} with {len(class_names)} classes.")

    # 5) Mel transform on **CPU** to avoid MPS STFT window mismatch
    mel_t = get_mel_transform(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )
    # (Do NOT move mel_t to device; keep it on CPU.)

    # 6) Run prediction
    predict_one(
        Path(args.wav),
        model,
        mel_t,
        device,
        class_names,
        sr=args.sr,
        topk=args.topk,
        out_dir=args.out_dir,
        duration=args.duration,
    )