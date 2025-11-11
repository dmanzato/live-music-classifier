
# live-music-classifier

Real-time music genre classification with **PyTorch**: microphone or dataset playback → **log-mel spectrogram** → **CNN** → **live Top-K predictions** (with a clean UI and keyboard controls).

[![CI](https://img.shields.io/github/actions/workflow/status/dmanzato/live-music-classifier/ci.yml?branch=main)](../../actions)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%F0%9F%AA%80-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**Repository**: [https://github.com/dmanzato/live-music-classifier](https://github.com/dmanzato/live-music-classifier)

> **Demos**  
> 
> **Live Streaming** (real-time microphone input):  
> ![live streaming inference](docs/demo_stream.gif)  
> 
> **Dataset Visualization** (interactive browser with rolling analysis):  
> ![dataset visualization](docs/demo_vis.gif)  

---

## Project Overview

This project provides a complete pipeline for **music genre classification**:

- **Training**: Train CNN models (SmallCNN or ResNet18) on **GTZAN** with optional data augmentation  
- **Inference**: Classify audio files with **Top-K predictions** and **spectrogram** visualization  
- **Live Streaming**: Real-time **microphone input** with **live spectrogram** and predictions  
- **Visualization**: Interactive dataset browser with ground truth **vs** predictions

### Key Features

- **Model Architectures**: SmallCNN (custom lightweight CNN) and ResNet18 (adapted for 1-channel input)  
- **Data Augmentation**: Optional SpecAugment (frequency & time masking)  
- **Live Inference**: Real-time microphone streaming with **trend line visualization**, **keyboard controls**, and **per-sample normalization**  
- **Visualizations**: Spectrograms with auto-gain, Top-K prediction bars with percentage labels, rolling trend lines, and interactive dataset browser  
- **Robust Audio I/O**: Uses `soundfile`/`sounddevice` (avoids torchcodec/FFmpeg dependency issues)  
- **Multi-device**: CPU, CUDA, and Apple Silicon **MPS**

### Project Structure

```
live-music-classifier/
├── train.py              # Main training script
├── predict.py            # Single-file inference
├── evaluate.py           # Model evaluation and comparison
├── demo_shapes.py        # Shape verification demo
├── setup.py              # Package setup and CLI entry points
├── requirements.txt      # Python dependencies
├── Makefile              # Convenient shortcuts for common tasks
├── models/
│   └── small_cnn.py      # CNN architecture definitions
├── datasets/
│   ├── gtzan.py          # GTZAN dataset loader
│   └── urbansound8k.py   # Legacy UrbanSound8K loader (from migration)
├── transforms/
│   └── audio.py          # Audio preprocessing & augmentation
├── utils/
│   ├── models.py         # Shared model building utilities
│   ├── class_map.py      # Class map loading/saving utilities
│   ├── device.py          # Device selection utilities (CPU/CUDA/MPS)
│   └── logging.py        # Logging configuration
├── scripts/
│   ├── stream_infer.py   # Live microphone streaming inference
│   ├── vis_dataset.py    # Interactive dataset browser with rolling analysis
│   ├── record_wav.py     # Audio recording utility
│   └── gen_demo_gif.py   # Demo GIF generator (stream & vis modes)
├── examples/             # Example scripts and tutorials
│   ├── train_example.sh
│   ├── inference_example.sh
│   ├── record_and_predict.sh
│   ├── quick_start.py
│   ├── live-music-classifier-tutorial.ipynb
│   └── README.md
├── tests/                # Test suite
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_transforms.py
│   ├── test_utils.py
│   └── README.md
├── docs/                 # Documentation assets
│   ├── demo_stream.gif   # Streaming inference demo
│   └── demo_vis.gif      # Dataset visualization demo
├── artifacts/            # Training outputs (models, confusion matrices, class_map.json)
└── pred_artifacts/       # Prediction outputs (spectrograms, predictions)
```

### Core Components

**Models** (`models/small_cnn.py`):
- `SmallCNN`: Lightweight 3-layer CNN for spectrogram classification
- Input: `[B, 1, n_mels, time]` log-mel spectrograms (normalized per-sample)
- Output: Class logits

**Dataset** (`datasets/gtzan.py`):
- `GTZAN`: PyTorch Dataset loader
- Handles audio loading, resampling, padding/trimming, and log-mel conversion
- Validates and skips corrupted audio files automatically
- Maps genre folders to contiguous indices (10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)

**Transforms** (`transforms/audio.py`):
- `get_mel_transform()`: Creates MelSpectrogram transform
- `wav_to_logmel()`: Converts waveform → log-mel spectrogram
- `SpecAugment`: Frequency and time masking augmentation (optional during training)

---

## Installation

### Option 1: Install as a package (recommended for CLI commands)

Install the package in development mode to get CLI entry points:

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install package and dependencies
pip install -e .

# Now you can use CLI commands:
live-music-train --help
live-music-predict --help
live-music-stream --help
```

This installs the package and makes the CLI commands (`live-music-train`, `live-music-predict`, `live-music-stream`) available system-wide in your virtual environment.

### Option 2: Install dependencies only

If you prefer to run scripts directly with Python:

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies only
pip install -r requirements.txt

# Run scripts with Python (requires PYTHONPATH=. or running from repo root)
python train.py --help
python predict.py --help
python scripts/stream_infer.py --help
```

---

## Quickstart

After installation, verify the setup:

```bash
# Verify shapes end-to-end
python demo_shapes.py
```

**Using CLI Commands** (if installed as package):
```bash
live-music-train --data_root /path/to/GTZAN --epochs 1 --model smallcnn
live-music-predict --wav /path/to/audio.wav --data_root /path/to/GTZAN
live-music-stream --list-devices
```

**Using Python directly** (if dependencies only):
```bash
python train.py --data_root /path/to/GTZAN --epochs 1 --model smallcnn
python predict.py --wav /path/to/audio.wav --data_root /path/to/GTZAN
python scripts/stream_infer.py --list-devices
```

> **Tip**: Want a one-liner developer flow? Use the included Makefile:
> `make setup`, `make train`, `make predict FILE=...`, `make stream DEVICE="..."`.

---

## Makefile Shortcuts

The project includes a `Makefile` with convenient shortcuts for common tasks. After running `make setup`, you can use:

```bash
# Setup virtual environment and install dependencies
make setup

# Train a model (uses default split ratios and parameters)
make train

# Run inference on a WAV file
make predict FILE=/path/to/audio.wav

# Start live streaming inference
make stream
# Or with a specific device:
make stream DEVICE="MacBook Pro Microphone"

# Visualize dataset samples
make vis
# Or with specific split:
make vis SPLIT=test

# Generate demo GIFs for README (creates both demo_stream.gif and demo_vis.gif)
make demo

# Run code quality checks
make lint      # Run ruff linter
make test      # Run pytest
make typecheck # Run mypy type checker
```

All Makefile targets activate the virtual environment automatically. See `Makefile` for default parameters and customization options.

---

## Train on GTZAN

> **Note**: If you installed the package (`pip install -e .`), you can use `live-music-train` instead of `python train.py`.

1) Download **GTZAN** and point `--data_root` to its root directory.

Expected layout:
```
GTZAN/
├── blues/
│   ├── blues.00000.wav
│   └── ...
├── classical/
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
```

2) Example run (default parameters: sr=22050, n_mels=128, n_fft=1024, hop_length=512):

```bash
python train.py \
    --data_root /path/to/GTZAN \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --batch_size 16 \
    --epochs 5 \
    --model smallcnn \
    --use_specaug \
    --sr 22050 \
    --n_mels 128 \
    --n_fft 1024 \
    --hop_length 512
```

Switch to ResNet18:

```bash
python train.py \
    --data_root /path/to/GTZAN \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --batch_size 16 \
    --epochs 5 \
    --model resnet18 \
    --sr 22050 \
    --n_mels 128 \
    --n_fft 1024 \
    --hop_length 512
```

After each epoch you'll get:

- Macro-F1 on validation  
- A saved confusion matrix under `artifacts/confusion_matrix_epochX.png`  
- Best model checkpoint: `artifacts/best_model.pt`  
- Class map file: `artifacts/class_map.json` (saved automatically for inference scripts)

---

## Inference on your own WAVs

> **Note**: If you installed the package (`pip install -e .`), you can use `live-music-predict` instead of `python predict.py`.

```bash
# Uses artifacts/best_model.pt by default
python predict.py \
    --wav /path/to/your_music.wav \
    --data_root /path/to/GTZAN \
    --model resnet18 \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 \
    --topk 5 \
    --out_dir pred_artifacts
```

- Saves `pred_artifacts/spectrogram.png`  
- Class names are loaded from `artifacts/class_map.json` (created automatically by `train.py`), or fallback to GTZAN genre folders if not found.

---

## Record audio from your mic (quick utility)

Install the extra deps (already listed in `requirements.txt`):

```bash
pip install -r requirements.txt
```

List audio devices:

```bash
python scripts/record_wav.py --list-devices
```

Record 30 seconds to WAV (mono, 22050Hz by default to match model training) - GTZAN uses 30-second clips:

```bash
python scripts/record_wav.py --out my_clip.wav --seconds 30 --sr 22050 --channels 1
```

Then run inference on what you recorded:

```bash
python predict.py --wav my_clip.wav   --data_root /path/to/GTZAN
```

**macOS**: If you get an input-permission error, go to *System Settings → Privacy & Security → Microphone* and allow Terminal/iTerm access.

---

## Streaming Inference (Live Microphone)

Real-time microphone input with **live spectrogram**, **Top-K predictions**, and **trend line visualization**.

```bash
# Using the CLI command (after pip install)
live-music-stream \
    --data_root /path/to/GTZAN \
    --checkpoint artifacts/best_model.pt \
    --model resnet18 \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 \
    --win_sec 7.5 --hop_sec 0.5 --topk 5 \
    --spec_auto_gain --spec_pmin 5 --spec_pmax 95

# Or using Python directly
python scripts/stream_infer.py \
    --data_root /path/to/GTZAN \
    --checkpoint artifacts/best_model.pt \
    --model resnet18 \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 \
    --win_sec 7.5 --hop_sec 0.5 --topk 5 \
    --spec_auto_gain --spec_pmin 5 --spec_pmax 95
```

**Features**

- **Rolling window buffer** (default 7.5 seconds) with configurable length  
- **Live spectrogram** with auto-gain color scaling (percentile-based)  
- **Top-K predictions** with horizontal bar chart and percentage labels  
- **Trend line visualization** showing top-1 probability over time (with optional EMA smoothing)  
- **Per-sample normalization** (mean/std) for improved model performance  
- **Keyboard controls**: `SPACE` to pause/resume, `q` to quit  
- **Configurable refresh rate** via `--hop_sec`

**Device Selection**

List available audio input devices:
```bash
live-music-stream --list-devices
# or
python scripts/stream_infer.py --list-devices
```

Use `--device <index>` or `--device '<substring>'` to pick a specific microphone.

**Important**: Make sure `--win_sec` matches the `--duration` used during training. If you trained with `--duration 30.0`, use `--win_sec 30.0` for streaming.

---

## Dataset Visualization

Interactive viewer for browsing dataset samples with **rolling analysis windows** and **trend line visualization**:

```bash
python scripts/vis_dataset.py \
    --data_root /path/to/GTZAN/genres_original \
    --split test \
    --checkpoint artifacts/best_model.pt \
    --model resnet18 \
    --sr 22050 --n_mels 128 --n_fft 1024 --hop_length 512 \
    --duration 30.0 --hold_sec 30.0 \
    --ana_win_sec 3.0 --ana_hop_sec 0.5 \
    --spec_auto_gain --spec_pmin 5 --spec_pmax 95 \
    --play_audio --topk 5
```

**Features**

- **Rolling analysis window**: Analyzes a sliding window (default 3.0s) over each audio clip as it plays  
- **Trend line**: Shows top-1 prediction probability over time for the current clip  
- **Top-K predictions** with percentage labels on horizontal bars  
- **Ground truth vs predictions**: Color-coded title (green=correct, red=incorrect)  
- **Audio playback**: Synchronized playback with rolling analysis updates  
- **Auto-gain spectrogram**: Percentile-based color scaling for better visualization

**Keyboard Controls**

- `Left/Right`: Previous/next sample  
- `Space`: Pause/resume auto-advance  
- `P`: Toggle audio playback  
- `Q`: Quit

---

## Scripts Overview

### Main Scripts

#### `train.py`
Main training script. Trains models on GTZAN with train/val/test splits. Supports both SmallCNN and ResNet18 with optional SpecAugment, mixup augmentation, class weight balancing, and per-sample spectrogram normalization for improved convergence.

#### `predict.py`
Single-file inference. Classifies a WAV file and outputs Top-K predictions with probabilities. Saves spectrogram visualization.

#### `evaluate.py`
Model evaluation and comparison script. Evaluates trained models on a test set and computes accuracy, F1 scores, and per-class metrics. Can compare multiple models side-by-side.

#### `demo_shapes.py`
Minimal shape verification script. Tests the pipeline end-to-end: waveform → log-mel → model to verify tensor shapes. Useful for debugging shape mismatches and verifying the preprocessing pipeline works correctly before training. Generates a synthetic sine wave, converts it to a log-mel spectrogram, and runs it through the model to confirm all tensor dimensions are correct.

### Scripts Directory (`scripts/`)

#### `scripts/stream_infer.py`
Live streaming inference from microphone. Real-time predictions with live spectrogram, Top-K prediction bars with percentage labels, and trend line visualization showing top-1 probability over time. Features per-sample normalization, auto-gain spectrogram scaling, keyboard controls (pause/resume, quit), and optional EMA smoothing for trend lines.

#### `scripts/vis_dataset.py`
Interactive dataset browser with rolling analysis windows. Shows ground truth vs predictions with color-coded feedback, rolling spectrogram analysis, trend line visualization, Top-K prediction bars with percentage labels, and synchronized audio playback. Supports keyboard navigation (previous/next, pause/resume, toggle playback).

#### `scripts/record_wav.py`
Audio recording utility. Records from microphone and saves as WAV file (mono, configurable sample rate, defaults to 22050Hz to match model training).

#### `scripts/gen_demo_gif.py`
Generates animated GIFs for the README matching the visual style of `stream_infer.py` and `vis_dataset.py`. Supports both "stream" mode (single-file continuous) and "vis" mode (multi-file browsing). Creates frame-by-frame animations with rolling spectrograms, Top-K prediction bars with percentage labels, and trend line visualizations.

---

## Examples

See the [examples/](examples/) directory for:
- Training scripts with different configurations
- Inference examples
- Record-and-predict workflows
- Jupyter notebook tutorials
- Complete usage documentation

---

## Why this repo?

Most GTZAN examples stop at "train offline & classify WAVs."  
This project also provides **real-time microphone streaming**, **live spectrogram visualization**, **keyboard controls**, and **dataset playback with synchronized audio**, all in a minimal, hackable codebase.

```
audio → STFT → Mel → Log → CNN → Top-K + UI
```

---

## Model Evaluation and Comparison

Compare model performance on a test set:

```bash
# Evaluate both models (requires trained checkpoints)
live-music-evaluate \
    --data_root /path/to/GTZAN \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --checkpoint_smallcnn artifacts/best_model_smallcnn.pt \
    --checkpoint_resnet18 artifacts/best_model_resnet18.pt \
    --output results.csv
```

Or evaluate a single model:

```bash
live-music-evaluate \
    --data_root /path/to/GTZAN \
    --models smallcnn \
    --checkpoint_smallcnn artifacts/best_model.pt
```

**Metrics reported:**
- **Accuracy**: Overall classification accuracy
- **Macro F1**: Average F1 score across all classes (unweighted)
- **Weighted F1**: Average F1 score weighted by class support
- **Per-class F1**: F1 score for each individual class
- **Model parameters**: Number of trainable parameters

The script outputs a comparison table when evaluating multiple models, showing which performs better.

**Note**: To compare both models, train each separately and save with different checkpoint names:
```bash
# Train SmallCNN
live-music-train --data_root /path/to/GTZAN --model smallcnn --epochs 10
mv artifacts/best_model.pt artifacts/best_model_smallcnn.pt

# Train ResNet18
live-music-train --data_root /path/to/GTZAN --model resnet18 --epochs 10
mv artifacts/best_model.pt artifacts/best_model_resnet18.pt

# Compare
live-music-evaluate --data_root /path/to/GTZAN \
    --checkpoint_smallcnn artifacts/best_model_smallcnn.pt \
    --checkpoint_resnet18 artifacts/best_model_resnet18.pt
```

---

## Testing

Run the test suite:

```bash
pytest tests/
```

For a coverage report:

```bash
pytest tests/ --cov=. --cov-report=html
```

See `tests/README.md` for more details.

---

## License

[MIT](LICENSE)
