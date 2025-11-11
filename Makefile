# Developer UX for live-music-classifier
# Usage:
#   make setup                      # create .venv + install deps
#   make train                      # train on GTZAN
#   make predict FILE=...           # run predict.py on a WAV
#   make stream [DEVICE=...]        # live mic inference
#   make vis [SPLIT=test]           # dataset visualizer (shuffled; 30s per song)
#   make demo                       # build README demo GIF (docs/demo.gif)
#   make lint test typecheck        # quality checks
#
# Tips:
#   Disable shuffle:      make vis NO_SHUFFLE=1
#   Change duration:      make vis DUR=20 HOLD=20
#   Tweak rolling window: make vis WIN=4 HOP=0.25
#   Change data root:     make DATA_ROOT=/path/to/GTZAN/genres_original vis

SHELL := /usr/bin/env bash
.ONESHELL:
PY := python                                  # Python interpreter to use
VENV := .venv                                 # Virtualenv path
ACT := source $(VENV)/bin/activate            # Activates the venv in this shell

# -------- Common defaults (override via make VAR=...) --------
# Paths
DATA_ROOT ?= ../data/GTZAN/genres_original     # Root folder of GTZAN dataset
CHECKPOINT ?= artifacts/best_model.pt          # Model checkpoint to load for inference/vis
OUT_DIR ?= pred_artifacts                      # Output dir for predict artifacts

# Model / feature extraction (training & inference must agree)
MODEL ?= resnet18                              # Backbone: resnet18 | smallcnn
SR ?= 22050                                    # Sample rate (Hz) used by the model
N_MELS ?= 128                                  # Number of mel bins
N_FFT ?= 1024                                  # STFT FFT size (samples)
HOPLEN ?= 512                                  # STFT hop length (samples)

# Training
EPOCHS ?= 5                                    # Default epochs for quick runs
BATCH ?= 16                                    # Training batch size
TRAIN_RATIO ?= 0.8                             # Train split ratio
VAL_RATIO ?= 0.1                               # Val split ratio (test is implied)

# Streaming / Visualization (script params)
DUR ?= 30.0                                    # Seconds to analyze/play per item
HOLD ?= $(DUR)                                 # Auto-advance dwell time (defaults to DUR)
TOPK ?= 5                                      # Top-K classes to show
WIN ?= 3.0                                     # Analysis window length (sec)
HOP ?= 0.5                                     # Analysis hop between updates (sec)

SPLIT ?= test                                  # Dataset split to visualize (train|val|test|all)

# Shuffle control for vis (default: shuffle ON; pass NO_SHUFFLE=1 to disable)
VIS_SHUFFLE_FLAG := $(if $(NO_SHUFFLE),--no_shuffle,)

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make setup"
	@echo "  make train"
	@echo "  make predict FILE=path/to.wav"
	@echo "  make stream [DEVICE=substring or index]"
	@echo "  make vis [SPLIT=test] [NO_SHUFFLE=1] [DUR=30 HOLD=30 WIN=3 HOP=0.5]"
	@echo "  make demo"
	@echo "  make lint | test | typecheck"

$(VENV)/bin/python:
	$(PY) -m venv $(VENV)

.PHONY: setup
setup: $(VENV)/bin/python
	$(ACT)
	pip install -U pip
	pip install -r requirements.txt
	# dev tools
	pip install ruff mypy pytest soundfile scipy
	@echo "✅ Setup complete. Activate with: source $(VENV)/bin/activate"

.PHONY: train
train:
	$(ACT)
	export PYTHONPATH=.
	$(PY) train.py \
	  --data_root "$(DATA_ROOT)" \
	  --train_ratio $(TRAIN_RATIO) \
	  --val_ratio $(VAL_RATIO) \
	  --batch_size $(BATCH) \
	  --epochs $(EPOCHS) \
	  --model $(MODEL) \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN)

.PHONY: predict
predict:
	@if [ -z "$(FILE)" ]; then echo "Usage: make predict FILE=path/to.wav"; exit 1; fi
	$(ACT)
	export PYTHONPATH=.
	$(PY) predict.py \
	  --wav "$(FILE)" \
	  --data_root "$(DATA_ROOT)" \
	  --checkpoint "$(CHECKPOINT)" \
	  --model $(MODEL) \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN) \
	  --topk $(TOPK) \
	  --out_dir "$(OUT_DIR)"

.PHONY: stream
stream:
	$(ACT)
	export PYTHONPATH=.
	$(PY) scripts/stream_infer.py \
	  --data_root "$(DATA_ROOT)" \
	  --checkpoint "$(CHECKPOINT)" \
	  --model $(MODEL) \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN) \
	  --win_sec $(WIN) --hop_sec $(HOP) \
	  --topk $(TOPK) \
	  $(if $(DEVICE),--device "$(DEVICE)",)

.PHONY: vis
vis:
	$(ACT)
	export PYTHONPATH=.
	$(PY) scripts/vis_dataset.py \
	  --data_root "$(DATA_ROOT)" \
	  --split $(SPLIT) \
	  --checkpoint "$(CHECKPOINT)" \
	  --model $(MODEL) \
	  --topk $(TOPK) \
	  --spec_auto_gain --spec_pmin 5 --spec_pmax 95 \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN) \
	  --duration $(DUR) --hold_sec $(HOLD) \
	  --ana_win_sec $(WIN) --ana_hop_sec $(HOP) \
	  --play_audio --out_latency high \
	  $(VIS_SHUFFLE_FLAG)

.PHONY: demo
demo:
	$(ACT)
	export PYTHONPATH=.
	$(PY) scripts/gen_demo_gif.py \
	  --data_root "$(DATA_ROOT)" \
	  --inputs "$(DATA_ROOT)/blues" \
	  --checkpoint "$(CHECKPOINT)" \
	  --model $(MODEL) \
	  --sr $(SR) --n_mels $(N_MELS) --n_fft $(N_FFT) --hop_length $(HOPLEN) \
	  --out docs/demo.gif \
	  --max_files 6

.PHONY: lint
lint:
	$(ACT)
	ruff check .

.PHONY: typecheck
typecheck:
	$(ACT)
	mypy audio_classify scripts || true

.PHONY: test
test:
	$(ACT)
	PYTHONPATH=. pytest -q