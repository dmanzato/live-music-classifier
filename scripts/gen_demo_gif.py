#!/usr/bin/env python
"""
Generate demo GIFs that match the visual style of vis_dataset.py and stream_infer.py:
- Rolling log-mel spectrogram (with optional auto-gain)
- Top-5 bar chart with percentage labels
- Top-1 probability trend line over time

USAGE EXAMPLES
--------------
# 1) Single demo GIF (multi-file "vis" style)
PYTHONPATH=. python scripts/gen_demo_gif.py \
  --data_root /path/to/GTZAN/genres_original \
  --inputs /path/to/GTZAN/genres_original/blues \
  --checkpoint artifacts/best_model.pt \
  --model resnet18 \
  --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 \
  --mode vis \
  --duration 12.0 --ana_win_sec 3.0 --ana_hop_sec 0.25 \
  --spec_auto_gain --spec_pmin 5 --spec_pmax 95 \
  --out docs/demo_vis.gif --max_files 6

# 2) Single demo GIF (single-file "stream" style)
PYTHONPATH=. python scripts/gen_demo_gif.py \
  --data_root /path/to/GTZAN/genres_original \
  --inputs /path/to/GTZAN/genres_original/jazz \
  --checkpoint artifacts/best_model.pt \
  --model resnet18 \
  --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 \
  --mode stream \
  --duration 12.0 --ana_win_sec 3.0 --ana_hop_sec 0.25 \
  --spec_auto_gain --spec_pmin 5 --spec_pmax 95 \
  --out docs/demo_stream.gif --max_files 1

# 3) Build BOTH GIFs in a single run (provide both outputs)
PYTHONPATH=. python scripts/gen_demo_gif.py \
  --data_root /path/to/GTZAN/genres_original \
  --inputs /path/to/GTZAN/genres_original \
  --checkpoint artifacts/best_model.pt \
  --model resnet18 \
  --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 \
  --duration 12.0 --ana_win_sec 3.0 --ana_hop_sec 0.25 \
  --spec_auto_gain --spec_pmin 5 --spec_pmax 95 \
  --out_vis docs/demo_vis.gif \
  --out_stream docs/demo_stream.gif \
  --max_files 6
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

# ---- Force a headless backend BEFORE importing pyplot ----
import matplotlib
matplotlib.use("Agg")  # headless backend; stable pixel buffer APIs

import matplotlib.pyplot as plt
from matplotlib import gridspec
import imageio.v2 as imageio
import soundfile as sf
from scipy.signal import resample_poly
from PIL import Image
# project imports
from transforms.audio import get_mel_transform, wav_to_logmel
from utils.device import get_device, get_device_name
from utils.models import build_model

# Default GTZAN labels (fallback if no class_map.json)
GTZAN_GENRES = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]

# -------------------- Helpers --------------------
def _read_json_classmap(p: Path) -> list[str]:
    with open(p, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "idx2name" in data:
        return data["idx2name"]
    if isinstance(data, list):
        return data
    raise ValueError("Invalid class_map.json format")

def _compute_spec_limits(mel_img: np.ndarray, auto_gain: bool, pmin: float, pmax: float, prev=None):
    if auto_gain:
        lo = np.percentile(mel_img, pmin)
        hi = np.percentile(mel_img, pmax)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            lo, hi = (mel_img.min(), mel_img.max())
            if lo == hi:
                hi = lo + 1e-6
        return lo, hi
    else:
        if prev is None:
            lo, hi = (mel_img.min(), mel_img.max())
            if lo == hi:
                hi = lo + 1e-6
            return lo, hi
        return prev

def _load_wav_centered(path: Path, target_sr: int, duration: float) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != target_sr:
        x = resample_poly(x, target_sr, sr).astype(np.float32)
        sr = target_sr
    N = int(duration * sr)
    if len(x) < N:
        pad = N - len(x)
        left, right = pad // 2, pad - pad // 2
        x = np.pad(x, (left, right))
    else:
        start = max(0, (len(x) - N) // 2)
        x = x[start:start + N]
    return x, sr

def _glob_wavs(root_or_dir: Path) -> list[Path]:
    root = Path(root_or_dir)
    if root.is_file() and root.suffix.lower() == ".wav":
        return [root]
    wavs = list(root.rglob("*.wav"))
    return sorted(wavs)

def _gt_from_path(p: Path) -> str | None:
    # For GTZAN, label is the parent dir name (genres_original/<label>/<file>.wav)
    try:
        return p.parent.name
    except Exception:
        return None

def _fig_to_rgb(fig):
    """
    Return an HxWx3 uint8 RGB array for any Matplotlib backend.
    IMPORTANT: we copy() the buffer so frames don't alias the same memory.
    """
    fig.canvas.draw()

    # Preferred: backend-agnostic RGBA buffer
    try:
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        rgb = rgba.reshape((h, w, 4))[:, :, :3]
        return rgb.copy()  # <-- CRITICAL
    except Exception:
        pass

    # Agg fallback
    try:
        buf, (w, h) = fig.canvas.print_to_buffer()
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        return rgba[:, :, :3].copy()  # <-- CRITICAL
    except Exception:
        pass

    # Classic RGB path
    try:
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape((h, w, 3)).copy()  # <-- CRITICAL
    except Exception:
        pass

    # macOS ARGB last resort
    try:
        w, h = fig.canvas.get_width_height()
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
        return argb[:, :, 1:4].copy()  # <-- CRITICAL
    except Exception as e:
        raise RuntimeError(f"Could not extract RGB buffer from canvas: {e}")

# -------------------- Core drawing --------------------
def _init_figure(n_mels: int, duration: float, topk: int) -> tuple[plt.Figure, dict]:
    plt.ioff()
    fig = plt.figure(figsize=(12, 7))
    outer = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.25, bottom=0.18, left=0.06, right=0.98, top=0.90)
    ax_spec = fig.add_subplot(outer[0, 0])

    right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], height_ratios=[3, 1], hspace=0.35)
    ax_bar   = fig.add_subplot(right[0, 0])
    ax_trend = fig.add_subplot(right[1, 0])

    # spectrogram placeholder (tiny noise so it's not a solid color)
    init_img = np.random.randn(n_mels, 64) * 1e-6
    im = ax_spec.imshow(init_img, origin="lower", aspect="auto")
    ax_spec.set_xlabel("Frames")
    ax_spec.set_ylabel("Mel bins")
    ax_spec.set_title("Spectrogram (rolling window)")

    # bars
    bars = ax_bar.barh(range(topk), np.zeros(topk), align="center")
    ax_bar.set_xlim(0.0, 1.0)
    ax_bar.set_yticks(range(topk))
    ax_bar.set_yticklabels([""] * topk)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Probability")
    ax_bar.set_title(f"Top-{topk} predictions")
    bar_texts = [ax_bar.text(0.0, i, "", va="center", ha="left", fontsize=9) for i in range(topk)]

    # trend line
    ax_trend.set_xlim(0.0, duration)
    ax_trend.set_ylim(0.0, 1.0)
    (trend_line,) = ax_trend.plot([], [], linewidth=2)
    trend_dot = ax_trend.plot([], [], marker="o")[0]
    ax_trend.set_xlabel("Time (s)")
    ax_trend.set_ylabel("Top-1 p")
    ax_trend.grid(True, alpha=0.3)
    ax_trend.set_title("Top-1 trend")

    return fig, dict(
        ax_spec=ax_spec, im=im,
        ax_bar=ax_bar, bars=bars, bar_texts=bar_texts,
        ax_trend=ax_trend, trend_line=trend_line, trend_dot=trend_dot
    )

def _update_viz(fig: plt.Figure,
                vis: dict,
                probs: np.ndarray,
                mel_img: np.ndarray,
                class_names: list[str],
                topk: int,
                spec_auto_gain: bool,
                spec_pmin: float,
                spec_pmax: float,
                last_clim: tuple[float,float] | None,
                pred_idx_gt: tuple[int | None, int | None]) -> tuple[float, tuple[float,float] | None]:
    """
    Update spectrogram, bars (with percentages), and title color by correctness. Return (top1, new_clim).
    """
    im = vis["im"]; ax_bar = vis["ax_bar"]; bars = vis["bars"]; bar_texts = vis["bar_texts"]

    order = np.argsort(probs)[::-1][:topk]
    top_probs = probs[order]
    top_names = [class_names[j] if j < len(class_names) else f"class_{j}" for j in order]

    # bars + percentage labels
    xmax = max(1.0, float(top_probs[0]) * 1.1)
    ax_bar.set_xlim(0.0, xmax)
    ax_bar.set_yticklabels(top_names)
    for k, (bar, p) in enumerate(zip(bars, top_probs)):
        w = float(p)
        bar.set_width(w)
        label = f"{100.0 * w:.1f}%"
        thresh_inside = 0.18 * xmax
        if w >= thresh_inside:
            x_text = max(0.0, w - 0.02 * xmax); ha, color = "right", "white"
        else:
            x_text = min(w + 0.02 * xmax, xmax * 0.98); ha, color = "left", "black"
        bar_texts[k].set_text(label)
        bar_texts[k].set_x(x_text); bar_texts[k].set_y(k)
        bar_texts[k].set_ha(ha);    bar_texts[k].set_color(color)

    # spectrogram intensity scaling
    vmin, vmax = _compute_spec_limits(mel_img, spec_auto_gain, spec_pmin, spec_pmax, last_clim)
    if spec_auto_gain:
        im.set_clim(vmin=vmin, vmax=vmax)
        last_clim = (vmin, vmax)
    im.set_data(mel_img)

    # title color by correctness if GT known (stream mode: gt_idx=None)
    pred_idx, gt_idx = pred_idx_gt
    pred_name = class_names[pred_idx] if pred_idx is not None and pred_idx < len(class_names) else ("?" if pred_idx is None else f"class_{pred_idx}")
    top1_prob = float(top_probs[0])
    title = f"Pred: {pred_name} ({100.0 * top1_prob:.1f}%)"

    if gt_idx is not None and 0 <= gt_idx < len(class_names):
        gt_name = class_names[gt_idx]
        title += f"  |  GT: {gt_name}"
        color = "green" if (pred_idx == gt_idx) else "red"
        fig.suptitle(title, color=color, fontsize=12)
    else:
        # stream mode / unknown GT → no color kwarg
        fig.suptitle(title, fontsize=12)

    return top1_prob, last_clim

# -------------------- Main logic --------------------
def main():
    ap = argparse.ArgumentParser(description="Generate demo GIFs for live-music-classifier visuals.")
    ap.add_argument("--data_root", type=str, required=True, help="Path to dataset root (e.g., GTZAN genres_original).")
    ap.add_argument("--inputs", type=str, required=True, help="Directory (or single WAV) to sample from.")
    ap.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt")
    ap.add_argument("--model", type=str, default="resnet18", choices=["smallcnn", "resnet18"])
    ap.add_argument("--topk", type=int, default=5)

    # feature/audio
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=512)

    # rolling analysis
    ap.add_argument("--duration", type=float, default=12.0, help="Seconds per clip to visualize.")
    ap.add_argument("--ana_win_sec", type=float, default=3.0, help="Rolling analysis window.")
    ap.add_argument("--ana_hop_sec", type=float, default=0.25, help="Step between rolling updates.")
    ap.add_argument("--fps", type=int, default=12, help="GIF frame rate (frames/sec).")

    # spectrogram scaling
    ap.add_argument("--spec_auto_gain", action="store_true")
    ap.add_argument("--spec_pmin", type=float, default=5.0)
    ap.add_argument("--spec_pmax", type=float, default=95.0)

    # sampling / ordering
    ap.add_argument("--max_files", type=int, default=6, help="Max files to include (vis mode).")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle file order (vis mode).")

    # output control
    ap.add_argument("--mode", type=str, default=None, choices=["vis", "stream"], help="If set, builds only this mode into --out.")
    ap.add_argument("--out", type=str, default=None, help="Output GIF path for the selected --mode.")
    ap.add_argument("--out_vis", type=str, default=None, help="If set, also create a 'vis' GIF here.")
    ap.add_argument("--out_stream", type=str, default=None, help="If set, also create a 'stream' GIF here.")

    args = ap.parse_args()

    # Validate output combination
    if args.mode and not args.out:
        print("error: --mode requires --out", file=sys.stderr)
        sys.exit(2)
    if not args.mode and not (args.out_vis or args.out_stream):
        print("error: provide either --mode/--out OR --out_vis/--out_stream", file=sys.stderr)
        sys.exit(2)

    device = get_device()
    print(f"Using device: {get_device_name()} ({device})")

    # Load class names
    class_map_path = Path("artifacts") / "class_map.json"
    if class_map_path.exists():
        try:
            class_names = _read_json_classmap(class_map_path)
        except Exception:
            class_names = GTZAN_GENRES
    else:
        class_names = GTZAN_GENRES
    num_classes = len(class_names)

    # Build model
    state = torch.load(args.checkpoint, map_location=device)
    state = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model = build_model(args.model, num_classes=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()

    # Mel transform (CPU)
    mel_t = get_mel_transform(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )

    # Collect WAVs
    in_dir = Path(args.inputs)
    all_wavs = _glob_wavs(in_dir)
    if not all_wavs:
        print(f"No WAV files found under: {in_dir}", file=sys.stderr)
        sys.exit(1)

    def predict_window(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute probs + mel image for a 1D float32 segment (model sr)."""
        seg_t = torch.from_numpy(seg).unsqueeze(0)  # [1, T]
        logmel = wav_to_logmel(seg_t, sr=args.sr, mel_transform=mel_t)
        # match viewer normalization
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
        with torch.no_grad():
            feats = logmel.unsqueeze(0).to(device)  # [1,1,n_mels,time]
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        mel_img = logmel.squeeze(0).cpu().numpy()
        return probs, mel_img

    def simulate_stream_frames(wav_path: Path) -> list[np.ndarray]:
        """
        Single-file 'stream' style:
        - Slide a window ending at t from 0..duration by ana_hop_sec
        - For each step, update viz and capture a frame
        NOTE: No GT label is shown in stream mode.
        """
        x, _ = _load_wav_centered(wav_path, args.sr, args.duration)
        topk = max(1, min(args.topk, num_classes))
        fig, vis = _init_figure(args.n_mels, args.duration, topk)
        frames: list[np.ndarray] = []
        last_clim = None

        # In stream mode, do NOT show GT (even if file path reveals genre)
        gt_idx = None

        trend_t: list[float] = []
        trend_p: list[float] = []

        times = np.arange(0.0, args.duration + 1e-6, args.ana_hop_sec)
        win_n = int(round(min(args.ana_win_sec, args.duration) * args.sr))

        for t in times:
            end_n = int(round(min(t, args.duration) * args.sr))
            start_n = max(0, end_n - win_n)
            seg = x[start_n:end_n]
            if len(seg) < win_n:
                seg = np.pad(seg, (win_n - len(seg), 0))
            probs, mel_img = predict_window(seg)

            order = np.argsort(probs)[::-1]
            pred_idx = int(order[0]) if probs.size else None

            top1, last_clim = _update_viz(
                fig, vis, probs, mel_img, class_names, topk,
                args.spec_auto_gain, args.spec_pmin, args.spec_pmax, last_clim,
                (pred_idx, gt_idx)  # gt_idx=None hides GT in title
            )

            # update trend
            vis["trend_line"].set_data(trend_t + [t], trend_p + [top1])
            vis["trend_dot"].set_data([t], [top1])
            trend_t.append(t); trend_p.append(top1)

            frames.append(_fig_to_rgb(fig))

        plt.close(fig)
        return frames

    def multi_vis_frames(wavs: list[Path]) -> list[np.ndarray]:
        """
        Multi-file 'vis' style:
        - For each file: simulate rolling updates across the clip and append frames.
        - Keeps same viz layout; title shows Pred | GT colorized.
        """
        if args.shuffle:
            rng = np.random.default_rng()
            wavs = wavs.copy()
            rng.shuffle(wavs)

        wavs = wavs[: args.max_files]
        topk = max(1, min(args.topk, num_classes))
        fig, vis = _init_figure(args.n_mels, args.duration, topk)
        frames: list[np.ndarray] = []
        last_clim = None

        for i, wp in enumerate(wavs):
            # reset trend for each new clip
            vis["trend_line"].set_data([], [])
            vis["trend_dot"].set_data([], [])
            trend_t: list[float] = []
            trend_p: list[float] = []

            x, _ = _load_wav_centered(wp, args.sr, args.duration)
            gt_name = _gt_from_path(wp)
            gt_idx = class_names.index(gt_name) if (gt_name in class_names) else None

            # first immediate frame (t=0)
            win_n = int(round(min(args.ana_win_sec, args.duration) * args.sr))
            seg0 = x[max(0, 0 - win_n):0]
            if len(seg0) < win_n:
                seg0 = np.pad(seg0, (win_n - len(seg0), 0))
            probs0, mel0 = predict_window(seg0)
            pred_idx0 = int(np.argsort(probs0)[::-1][0]) if probs0.size else None
            top10, last_clim = _update_viz(
                fig, vis, probs0, mel0, class_names, topk,
                args.spec_auto_gain, args.spec_pmin, args.spec_pmax, last_clim,
                (pred_idx0, gt_idx)
            )
            vis["trend_line"].set_data([0.0], [top10])
            vis["trend_dot"].set_data([0.0], [top10])
            trend_t = [0.0]; trend_p = [top10]
            frames.append(_fig_to_rgb(fig))

            # rolling frames
            times = np.arange(args.ana_hop_sec, args.duration + 1e-6, args.ana_hop_sec)
            for t in times:
                end_n = int(round(min(t, args.duration) * args.sr))
                start_n = max(0, end_n - win_n)
                seg = x[start_n:end_n]
                if len(seg) < win_n:
                    seg = np.pad(seg, (win_n - len(seg), 0))

                probs, mel_img = predict_window(seg)
                pred_idx = int(np.argsort(probs)[::-1][0]) if probs.size else None
                top1, last_clim = _update_viz(
                    fig, vis, probs, mel_img, class_names, topk,
                    args.spec_auto_gain, args.spec_pmin, args.spec_pmax, last_clim,
                    (pred_idx, gt_idx)
                )

                vis["trend_line"].set_data(trend_t + [t], trend_p + [top1])
                vis["trend_dot"].set_data([t], [top1])
                trend_t.append(t); trend_p.append(top1)

                frames.append(_fig_to_rgb(fig))

        plt.close(fig)
        return frames


    def write_gif(frames: list[np.ndarray], out_path: Path, fps: int):
        """
        Animated GIF writer using Pillow.
        - Forces palette frames (GIF-native).
        - Copies data to avoid canvas buffer aliasing.
        - Disables optimization (prevents frame merging).
        """
        if not frames:
            raise ValueError("No frames to write")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy arrays before converting to PIL (defensive against aliasing)
        frames = [f.copy() for f in frames]

        rgb_imgs = [Image.fromarray(f, mode="RGB") for f in frames]
        pal_imgs = [im.copy().convert("P", palette=Image.ADAPTIVE, colors=256) for im in rgb_imgs]

        duration_ms = int(round(1000.0 / max(1, fps)))

        first = pal_imgs[0]
        rest  = pal_imgs[1:] if len(pal_imgs) > 1 else []

        first.save(
            str(out_path),
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            disposal=2,     # restore to background before next frame
            optimize=False, # do NOT merge identical frames
        )
        print(f"Saved demo GIF → {out_path}  ({len(pal_imgs)} frames @ {fps} fps)")

    # Build outputs as requested
    if args.mode:
        # Single mode → single output (--out)
        wavs = _glob_wavs(Path(args.inputs))
        if args.mode == "stream":
            # pick first WAV
            src = wavs[0]
            frames = simulate_stream_frames(src)
        else:
            frames = multi_vis_frames(wavs)
        write_gif(frames, Path(args.out), args.fps)
    else:
        # Possibly build both
        wavs = _glob_wavs(Path(args.inputs))
        if args.out_stream:
            src = wavs[0]
            frames = simulate_stream_frames(src)
            write_gif(frames, Path(args.out_stream), args.fps)
        if args.out_vis:
            frames = multi_vis_frames(wavs)
            write_gif(frames, Path(args.out_vis), args.fps)

if __name__ == "__main__":
    main()