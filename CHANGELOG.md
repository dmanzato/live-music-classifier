# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

