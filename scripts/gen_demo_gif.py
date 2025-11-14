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
from collections import deque
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

def _load_wav_centered(path: Path, target_sr: int, duration: float, loop: bool = False) -> tuple[np.ndarray, int]:
    """
    Load WAV file and center/crop or loop to match duration.
    
    Args:
        path: Path to WAV file
        target_sr: Target sample rate
        duration: Desired duration in seconds
        loop: If True and audio is shorter than duration, loop it instead of padding with zeros
    """
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != target_sr:
        x = resample_poly(x, target_sr, sr).astype(np.float32)
        sr = target_sr
    N = int(duration * sr)
    if len(x) < N:
        if loop:
            # Loop the audio to fill the required duration
            num_loops = int(np.ceil(N / len(x)))
            x = np.tile(x, num_loops)[:N]
        else:
            # Pad with zeros (original behavior)
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
def _init_figure(n_mels: int, duration: float, topk: int, win_sec: float, mode: str) -> tuple[plt.Figure, dict]:
    """
    Initialize figure with layout matching stream_infer.py (stream mode) or vis_dataset.py (vis mode).
    
    Args:
        n_mels: Number of mel bins
        duration: Total duration for vis mode
        topk: Number of top predictions to show
        win_sec: Window size in seconds (for stream mode negative time axis)
        mode: "stream" or "vis"
    """
    plt.ioff()
    # Match figure sizes: stream_infer.py uses (12, 6), vis_dataset.py uses (12, 7)
    fig = plt.figure(figsize=(12, 6) if mode == "stream" else (12, 7))
    outer = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.25, bottom=0.18, left=0.06, right=0.98, top=0.90)
    ax_spec = fig.add_subplot(outer[0, 0])

    right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], height_ratios=[3, 1], hspace=0.35)
    ax_bar   = fig.add_subplot(right[0, 0])
    ax_trend = fig.add_subplot(right[1, 0])

    # spectrogram placeholder (tiny noise so it's not a solid color)
    init_img = np.random.randn(n_mels, 64) * 1e-6
    
    if mode == "stream":
        # Stream mode: negative time axis (past → now at 0), matching stream_infer.py
        im = ax_spec.imshow(
            init_img, origin="lower", aspect="auto",
            extent=[-win_sec, 0.0, 0.0, float(n_mels)],
            cmap='magma',
            interpolation='nearest'
        )
        ax_spec.set_xlabel("Time (s, past → now)")
        ax_spec.set_ylabel("Mel bins")
        ax_spec.set_title("Spectrogram")
        ax_spec.set_xlim(-win_sec, 0.0)
    else:
        # Vis mode: frames axis, matching vis_dataset.py
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
    if mode == "stream":
        # Stream mode: negative time axis (past → now)
        # Fixed range [-60, 0] with 0 on right-hand side (matching user requirement)
        ax_trend.set_xlim(-60.0, 0.0)  # Fixed range: 0 on right-hand side y-axis
        ax_trend.set_xlabel("Time (s, past → now)")
        # Set ticks matching stream_infer.py: -50, -40, -30, -20, -10, 0 (NO -60)
        # stream_infer.py calculates: total_span = abs(xs[0]) where xs[0] ≈ -59.5, then
        # ticks = np.arange(-np.floor(59.5/10)*10, 10, 10) = np.arange(-50, 10, 10) = [-50, -40, -30, -20, -10]
        # then adds 0.0 if not present: [-50, -40, -30, -20, -10, 0]
        # We use xlim [-60, 0] but calculate ticks the same way (based on span, not xlim)
        step = 10.0
        # Calculate ticks the same way as stream_infer.py (using span, not xlim)
        # With x-axis range [-60, 0], the span is 60, but we calculate ticks as if span is ~59.5
        # to match stream_infer.py behavior (which uses xs[0] ≈ -59.5)
        total_span = 59.5  # Match stream_infer.py's xs[0] value
        ticks = np.arange(-np.floor(total_span / step) * step, step, step)  # [-50, -40, -30, -20, -10]
        if 0.0 not in ticks:
            ticks = np.append(ticks, 0.0)  # [-50, -40, -30, -20, -10, 0]
        ax_trend.set_xticks(ticks)
        # Set xlim to [-60, 0] (fixed range, but ticks don't include -60, matching stream_infer.py)
        ax_trend.set_xlim(-60.0, 0.0)
    else:
        # Vis mode: positive time from 0 to duration
        ax_trend.set_xlim(0.0, duration)
        ax_trend.set_xlabel("Time (s)")
    
    ax_trend.set_ylim(0.0, 1.0)
    (trend_line,) = ax_trend.plot([], [], linewidth=2)
    trend_dot = ax_trend.plot([], [], marker="o")[0]
    ax_trend.set_ylabel("Top-1 p")
    ax_trend.grid(True, alpha=0.25 if mode == "stream" else 0.3)
    # Stream mode: no title (matching stream_infer.py)
    # Vis mode: has title (matching vis_dataset.py)
    if mode != "stream":
        ax_trend.set_title("Top-1 trend")

    return fig, dict(
        ax_spec=ax_spec, im=im,
        ax_bar=ax_bar, bars=bars, bar_texts=bar_texts,
        ax_trend=ax_trend, trend_line=trend_line, trend_dot=trend_dot,
        mode=mode, win_sec=win_sec,
        first_spec_frame=True  # Track first frame for clim update
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
    Update spectrogram, bars (with percentages), and title. Return (top1, new_clim).
    Matches title format from stream_infer.py (stream mode) or vis_dataset.py (vis mode).
    """
    im = vis["im"]; ax_bar = vis["ax_bar"]; bars = vis["bars"]; bar_texts = vis["bar_texts"]
    mode = vis.get("mode", "vis")
    win_sec = vis.get("win_sec", 15.0)

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

    # spectrogram intensity scaling - matching stream_infer.py order
    vmin, vmax = _compute_spec_limits(mel_img, spec_auto_gain, spec_pmin, spec_pmax, last_clim)
    
    # Update spectrogram data FIRST (matching stream_infer.py order)
    im.set_data(mel_img)
    
    # Update spectrogram extent for stream mode (negative time axis)
    if mode == "stream":
        # Get n_mels from mel_img shape
        n_mels = mel_img.shape[0] if len(mel_img.shape) >= 2 else 128
        # Always show the most recent win_sec window: [-win_sec, 0.0]
        # This ensures continuous scrolling as new data arrives
        im.set_extent([-win_sec, 0.0, 0.0, float(n_mels)])
        vis["ax_spec"].set_xlim(-win_sec, 0.0)
    
    # Update color limits - matching stream_infer.py: only if auto_gain OR first frame
    first_frame = vis.get("first_spec_frame", True)
    if spec_auto_gain or first_frame:
        im.set_clim(vmin=vmin, vmax=vmax)
        last_clim = (vmin, vmax)
        vis["first_spec_frame"] = False

    # title format matching stream_infer.py (stream) or vis_dataset.py (vis)
    pred_idx, gt_idx = pred_idx_gt
    top1_prob = float(top_probs[0])
    
    if mode == "stream":
        # Stream mode: format like stream_infer.py - "{label} ({prob:4.1f}%)"
        pred_name = class_names[pred_idx] if pred_idx is not None and pred_idx < len(class_names) else ("?" if pred_idx is None else f"class_{pred_idx}")
        title = f"{pred_name} ({top1_prob*100:4.1f}%)"
        fig.suptitle(title, fontsize=12)
    else:
        # Vis mode: format like vis_dataset.py - "Pred: {name}  |  GT: {name}" with color
        pred_name = class_names[pred_idx] if pred_idx is not None and pred_idx < len(class_names) else ("?" if pred_idx is None else f"class_{pred_idx}")
        if gt_idx is not None and 0 <= gt_idx < len(class_names):
            gt_name = class_names[gt_idx]
            title = f"Pred: {pred_name}  |  GT: {gt_name}"
            color = "green" if (pred_idx == gt_idx) else "red"
            fig.suptitle(title, color=color, fontsize=12)
        else:
            title = f"Pred: {pred_name}"
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
        # Store raw mel for visualization (before normalization) - matching stream_infer.py
        mel_img_raw = logmel.squeeze(0).cpu().numpy()
        # Normalize for model inference
        logmel_norm = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
        with torch.no_grad():
            feats = logmel_norm.unsqueeze(0).to(device)  # [1,1,n_mels,time]
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        # Return raw mel for visualization (matching stream_infer.py behavior)
        return probs, mel_img_raw

    def simulate_stream_frames(wav_path: Path) -> list[np.ndarray]:
        """
        Single-file 'stream' style:
        - Slide a window ending at t from 0..duration by ana_hop_sec
        - For each step, update viz and capture a frame
        - Uses negative time axis (past → now) matching stream_infer.py
        - Fixed trend x-axis range [-60, 0] with 0 on right-hand side
        NOTE: No GT label is shown in stream mode.
        """
        # Load audio and loop it if needed to fill the full duration (90s)
        # This ensures continuous audio throughout warmup and capture periods
        x, _ = _load_wav_centered(wav_path, args.sr, args.duration, loop=True)
        topk = max(1, min(args.topk, num_classes))
        win_sec = args.ana_win_sec
        fig, vis = _init_figure(args.n_mels, args.duration, topk, win_sec, mode="stream")
        frames: list[np.ndarray] = []
        last_clim = None

        # In stream mode, do NOT show GT (even if file path reveals genre)
        gt_idx = None

        # For stream mode, use negative time (past → now at 0)
        # Build trend buffer similar to stream_infer.py with fixed range [-60, 0]
        # trend_len=120 with hop_sec=0.5 gives 120 points from -59.5 to 0.0
        # But we want fixed x-axis range [-60, 0] with 0 on the right-hand side
        trend_len = 120  # Match stream_infer.py default
        trend_buf = deque([np.nan] * trend_len, maxlen=trend_len)
        # Calculate x-axis points: with hop_sec=0.5, we get [-59.5, -59.0, ..., -0.5, 0.0]
        # This gives 120 points, but xlim is fixed to [-60, 0] so 0 appears on the right
        trend_xs = -np.arange(trend_len - 1, -1, -1, dtype=float) * args.ana_hop_sec

        times = np.arange(0.0, args.duration + 1e-6, args.ana_hop_sec)
        win_n = int(round(min(args.ana_win_sec, args.duration) * args.sr))

        # Calculate warm-up period: 90s to ensure trend buffer is fully populated
        # With trend_len=120 and hop_sec=0.5, we need 60s to fill completely, but 90s ensures stability
        # Then capture for 30s after warmup (total 120s duration)
        warmup_seconds = 90.0  # Extended warmup for stable trend buffer
        capture_seconds = 30.0  # Capture for 30s after warmup
        warmup_frames = int(warmup_seconds / args.ana_hop_sec) if args.ana_hop_sec > 0 else 0
        capture_frames = int(capture_seconds / args.ana_hop_sec) if args.ana_hop_sec > 0 else 60
        frame_count = 0

        for t in times:
            end_n = int(round(min(t, args.duration) * args.sr))
            start_n = max(0, end_n - win_n)
            seg = x[start_n:end_n]
            if len(seg) < win_n:
                seg = np.pad(seg, (win_n - len(seg), 0))
            probs, mel_img = predict_window(seg)

            order = np.argsort(probs)[::-1]
            pred_idx = int(order[0]) if probs.size else None

            frame_count += 1
            
            # Only update visualization and capture frames after warm-up period
            # This ensures the spectrogram shows fresh data when capture starts
            if warmup_frames < frame_count <= warmup_frames + capture_frames:
                # Reset first_spec_frame flag when we start capturing to ensure clim is set correctly
                if frame_count == warmup_frames + 1:
                    vis["first_spec_frame"] = True
                    last_clim = None  # Reset clim to ensure fresh calculation
                
                top1, last_clim = _update_viz(
                    fig, vis, probs, mel_img, class_names, topk,
                    args.spec_auto_gain, args.spec_pmin, args.spec_pmax, last_clim,
                    (pred_idx, gt_idx)  # gt_idx=None hides GT in title
                )

                # Update trend with negative time (past → now)
                # Append newest value (most recent at x=0, older points go negative)
                trend_buf.append(top1)
                
                # Use fixed x-axis range [-60, 0] - don't update limits dynamically
                # The x-axis is already fixed in _init_figure, just update the data
                y_data = np.array(list(trend_buf), dtype=float)
                vis["trend_line"].set_data(trend_xs, y_data)
                vis["trend_dot"].set_data([0.0], [top1])  # Current point at x=0

                frames.append(_fig_to_rgb(fig))
            else:
                # During warmup, still update trend buffer but don't update visualization
                # This ensures the trend graph is ready when capture starts
                order = np.argsort(probs)[::-1]
                pred_idx_temp = int(order[0]) if probs.size else None
                top1_temp = float(probs[order[0]]) if probs.size else 0.0
                trend_buf.append(top1_temp)

        plt.close(fig)
        return frames

    def multi_vis_frames(wavs: list[Path]) -> list[np.ndarray]:
        """
        Multi-file 'vis' style:
        - For each file: simulate rolling updates across the clip and append frames.
        - Keeps same viz layout; title shows Pred | GT colorized.
        - Uses positive time axis (0 to duration) matching vis_dataset.py
        - Captures full cycle from 0 to duration (full x-axis span of trend graph)
        """
        if args.shuffle:
            rng = np.random.default_rng()
            wavs = wavs.copy()
            rng.shuffle(wavs)

        wavs = wavs[: args.max_files]
        topk = max(1, min(args.topk, num_classes))
        win_sec = args.ana_win_sec
        # For vis mode, use duration=30.0 to span full trend graph (0 to 30s)
        vis_duration = 30.0  # Full cycle matching vis_dataset.py default
        fig, vis = _init_figure(args.n_mels, vis_duration, topk, win_sec, mode="vis")
        frames: list[np.ndarray] = []
        last_clim = None

        # Limit TOTAL frames to 61 (30s cycle: 1 frame at t=0 + 60 frames from 0.5s to 30s)
        max_total_frames = 61
        total_frame_count = 0

        for i, wp in enumerate(wavs):
            if total_frame_count >= max_total_frames:
                break  # Stop if we've reached the total frame limit
                
            # reset trend for each new clip
            vis["trend_line"].set_data([], [])
            vis["trend_dot"].set_data([], [])
            trend_t: list[float] = []
            trend_p: list[float] = []

            # Load full 30s clip for vis mode
            x, _ = _load_wav_centered(wp, args.sr, vis_duration)
            gt_name = _gt_from_path(wp)
            gt_idx = class_names.index(gt_name) if (gt_name in class_names) else None

            # first immediate frame (t=0)
            win_n = int(round(min(args.ana_win_sec, vis_duration) * args.sr))
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
            total_frame_count += 1
            
            if total_frame_count >= max_total_frames:
                break  # Stop if we've reached the total frame limit

            # rolling frames: capture only until we reach 61 total frames
            times = np.arange(args.ana_hop_sec, vis_duration + 1e-6, args.ana_hop_sec)
            
            for t in times:
                if total_frame_count >= max_total_frames:
                    break  # Stop after 61 total frames (30s cycle)
                    
                end_n = int(round(min(t, vis_duration) * args.sr))
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
                total_frame_count += 1
                
                if total_frame_count >= max_total_frames:
                    break  # Stop after 61 total frames

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
    # Ensure both GIFs have similar frame counts (~60 frames at 12 fps = ~5 seconds)
    # For stream mode: use duration that gives ~60 frames with ana_hop_sec
    # For vis mode: use 30.0s duration with ana_hop_sec=0.5 gives 60 frames
    if args.mode:
        # Single mode → single output (--out)
        wavs = _glob_wavs(Path(args.inputs))
        if args.mode == "stream":
            # pick first WAV
            # Adjust duration: 90s warmup + 30s capture = 120s total
            warmup_seconds = 90.0
            capture_seconds = 30.0
            stream_duration = warmup_seconds + capture_seconds
            # Temporarily override args.duration for stream mode
            original_duration = args.duration
            args.duration = stream_duration
            src = wavs[0]
            frames = simulate_stream_frames(src)
            args.duration = original_duration
        else:
            frames = multi_vis_frames(wavs)
        write_gif(frames, Path(args.out), args.fps)
    else:
        # Possibly build both
        wavs = _glob_wavs(Path(args.inputs))
        if args.out_stream:
            src = wavs[0]
            # Adjust duration: 90s warmup + 30s capture = 120s total
            warmup_seconds = 90.0
            capture_seconds = 30.0
            stream_duration = warmup_seconds + capture_seconds
            original_duration = args.duration
            args.duration = stream_duration
            frames = simulate_stream_frames(src)
            args.duration = original_duration
            write_gif(frames, Path(args.out_stream), args.fps)
        if args.out_vis:
            frames = multi_vis_frames(wavs)
            write_gif(frames, Path(args.out_vis), args.fps)

if __name__ == "__main__":
    main()