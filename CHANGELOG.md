# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2025-01-13

### Changed
- Updated `gen_demo_gif.py` to match UI of `stream_infer.py` and `vis_dataset.py`:
  - Stream mode: negative time axis (past → now), no trend title, fixed x-axis range [-60, 0] with ticks [-50, -40, -30, -20, -10, 0]
  - Vis mode: positive time axis, trend title, proper title format with GT comparison
  - Both modes: matching figure sizes, grid alpha, and spectrogram styling
- Updated `Makefile` demo targets with optimized parameters for GIF generation
- Updated `gen_demo_gif.py` to use raw mel spectrogram (before normalization) for visualization, matching `stream_infer.py`

### Fixed
- Fixed `gen_demo_gif.py` demo_vis.gif to capture exactly 61 frames (30s cycle) instead of multiple files × 61 frames
- Fixed `gen_demo_gif.py` demo_stream.gif spectrogram fade-out issue by looping audio instead of padding with zeros
- Fixed `gen_demo_gif.py` demo_stream.gif warmup period to 60s (fills entire trend x-axis) with 30s capture period (total 90s)
- Fixed `gen_demo_gif.py` demo_stream.gif trend x-axis to show fixed range [-60, 0] with 0 on right-hand side (matching `stream_infer.py`)
- Fixed `gen_demo_gif.py` demo_stream.gif trend x-axis ticks to exclude -60 label (matching `stream_infer.py`)
- Fixed `test_dataset.py` to match current GTZAN API (split values, sample rate defaults, and attribute names)

## [0.2.1] - 2025-01-13

### Changed
- Refactored `normalize_per_sample` function from `train.py` and `evaluate.py` into shared utility module `utils/normalization.py`
- Updated `evaluate.py` default parameters to match `train.py` (sr=22050, n_mels=128, hop_length=512, duration=5.0)
- Updated `evaluate.py` to use same split logic as `train.py` (random_split with seed=42)
- Updated streaming inference default window size from 7.5s to 15.0s
- Updated streaming inference examples to include `--spec_auto_gain` and `--auto_gain_norm` flags

### Removed
- Removed legacy `datasets/urbansound8k.py` file (no longer used after migration to GTZAN)
- Removed debug script `debug_model.py`

### Fixed
- Fixed version inconsistency: `setup.py` now correctly shows version 0.2.1
- Fixed `evaluate.py` to use per-sample normalization matching training pipeline
- Fixed documentation inconsistencies in examples and README

### Added
- Added comprehensive tests for `normalize_per_sample` utility function

## [0.2.0] - 2025-01-09

### Changed
- Converted from live-audio-classifier to live-music-classifier
- Replaced UrbanSound8K dataset with GTZAN dataset for music genre classification
- Changed from fold-based splits to train/val/test ratio-based splits
- Updated all CLI commands from `live-audio-*` to `live-music-*`
- Updated default audio duration from 4 seconds to 30 seconds (GTZAN standard)

### Added
- GTZAN dataset loader with genre folder-based structure
- Support for 10 music genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- Train/val/test split configuration with customizable ratios

## [0.1.0] - 2025-11-08

### Added
- Initial release of live-audio-classifier
- Support for SmallCNN and ResNet18 model architectures
- UrbanSound8K dataset loader with fold-based train/val splits
- Training script with optional SpecAugment data augmentation
- Single-file inference script with spectrogram visualization
- Live streaming inference from microphone with real-time predictions
- Interactive dataset visualization tool with keyboard controls
- Audio recording utility for quick WAV file capture
- Log-mel spectrogram preprocessing pipeline
- Support for CPU, CUDA, and MPS (Apple Silicon) devices
- Confusion matrix visualization during training
- Top-K prediction outputs with probabilities

### Features
- Real-time microphone streaming with EMA-smoothed predictions
- Live spectrogram visualization with auto-scaling
- Interactive dataset browser with audio playback
- Robust audio I/O using soundfile/sounddevice
- Comprehensive command-line interface for all scripts
- CLI entry points (`live-music-train`, `live-music-predict`, `live-music-stream`)
- Device listing utility (`--list-devices` flag)
- Continuous Integration (CI) workflow with automated testing
- Comprehensive test suite with 90%+ coverage on core modules

