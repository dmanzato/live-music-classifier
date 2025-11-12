# Roadmap

This document outlines potential improvements to the live-music-classifier project, organized by effort level and impact.

## Quick Wins (no retrain or tiny retrain)

### 1. Longer context in streaming (no code churn)

- **Bump `--win_sec` to 12–20s** (start with 15s) and keep `--hop_sec` at 0.5s.
- This gives the model longer harmonic/rhythmic cues (genre ≫ "short event").
- **Command:**
  ```bash
  PYTHONPATH=. python scripts/stream_infer.py ... --win_sec 15 --hop_sec 0.5 --spec_auto_gain --spec_pmin 5 --spec_pmax 95
  ```

### 2. Temporal smoothing on probabilities (already partially there)

- Turn on a light EMA for the top-1 series: `--trend_ema 0.3` (or 0.2–0.5).
- Also maintain a short average of the full probability vector over the last N hops (say last 4–6) before picking top-k. (If you want, I'll patch to keep a deque of prob vectors and average them.)

### 3. Multi-window ensemble at inference (no retrain)

- For each hop, compute logits with multiple window lengths (e.g., 5s, 10s, 15s) over the same right-aligned audio and average the softmaxes.
- This is cheap and very effective for music structure variance.

### 4. Mel front-end tweaks (no model change)

- Use richer time-freq resolution for music:
  - `--n_fft 2048` (or 4096 if you can afford it)
  - `--hop_length 512` (or 1024 for more smoothing)
  - `--n_mels 128` or 256
  - Set mel f-bands if you add args: `f_min≈30`, `f_max=sr/2`.
- Optionally switch from log-mel to **PCEN** (per-channel energy normalization) in your transform for better loudness robustness (can do at inference, but best if matched in training).

### 5. Calibration & class priors

- Do temperature scaling on the validation set (tiny step): fit a single temperature T and apply at inference to make the softmax less overconfident → more stable smoothing.
- If your stream domain skews (e.g., modern pop/live radio), you can apply a light prior reweight (logit bias) to temper frequent confusions.

---

## Medium lift (small code + retrain)

### 6. Train on longer clips + multi-crop

- Train with `--duration 15` (or 20–30s) but with random right-aligned crops each batch so the model sees many within-song contexts.
- Keep SpecAugment; consider Mixup on spectrograms (SpecMix) at α≈0.2–0.4.

### 7. Artist-conditional split + more robust metrics

- Ensure your GTZAN split avoids artist leakage (genre models often overfit timbre/production).
- Track per-genre F1—improves decisions on bias/augmentation.

### 8. Feature augmentations for music

- Add small time-stretch (±5–8%), pitch-shift (±1–2 semitones), small EQ bumps, and light room IR.
- Keep them realistic (don't break key/rhythm entirely).

### 9. CRNN head on top of 2D CNN

- Replace the pure 2D classifier head with a CNN → BiGRU/LSTM → FC.
- You keep 2D convs for timbre; the RNN learns temporal genre cues (groove/form).
- Streaming: maintain RNN hidden state between hops for faster, more stable updates.

### 10. Better normalization

- Use frequency-wise CMVN (cepstral mean/variance over each mel bin) or PCEN in both train + infer. You're already normalizing per-window; stabilize it to match train stats.

---

## Bigger leaps (dataset & model)

### 11. Larger / cleaner datasets

- GTZAN is tiny and has known issues. Graduate to **FMA** (Free Music Archive) (small/medium), **MTG-Jamendo** (tag→genre mapping), or similar.
- Use artist-disjoint splits. Even a few extra hours of data helps a lot.

### 12. Pretrained audio backbones

- Drop-in **PANNs CNN14**, **PaSST**, **AST/HTS-AT** pretrained on AudioSet, then fine-tune for your 10 genres.
- This is the single biggest accuracy lift per unit of effort.

### 13. Two-timescale streaming

- Run a fast head (short window, e.g., 3–5s) every hop for responsiveness + a slow head (15–30s) every few seconds. Fuse (average logits) with higher weight to the slow head.
- Feels much "stickier" on sustained genres but still reacts to changes.

---

## Recommended Implementation Order

What I'd do first (order of impact vs effort):

1. **(A)** Bump streaming window to 15s, keep 0.5s hop, add `--trend_ema 0.3`.
2. **(B)** Multi-window ensemble at inference: 5s + 10s + 15s averaged.
3. **(C)** Front-end: `n_fft=2048`, `hop=512`, `n_mels=128/256`, PCEN if quick to add.
4. **(D)** Longer-duration training (15–20s) with multi-crop + SpecAugment.
5. **(E)** When you're ready: swap backbone to a pretrained AST/PANNs and fine-tune.

