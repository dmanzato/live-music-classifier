# Developer UX for live-music-classifier
# Usage:
#   make setup                  # create .venv + install deps
#   make train                  # train on GTZAN
#   make predict FILE=...       # run predict.py on a WAV
#   make stream [DEVICE=...]    # live mic inference
#   make vis [SPLIT=test]       # dataset visualizer
#   make demo                   # build README demo GIF (docs/demo.gif)
#   make lint test typecheck    # basic quality checks

SHELL := /usr/bin/env bash
.ONESHELL:
PY := python
VENV := .venv
ACT := source $(VENV)/bin/activate

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make setup"
	@echo "  make train"
	@echo "  make predict FILE=path/to.wav"
	@echo "  make stream [DEVICE=substring or index]"
	@echo "  make vis [SPLIT=test]"
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
	  --data_root ../data/GTZAN \
	  --train_ratio 0.8 \
	  --val_ratio 0.1 \
	  --batch_size 16 \
	  --epochs 5 \
	  --model smallcnn

.PHONY: predict
predict:
	@if [ -z "$(FILE)" ]; then echo "Usage: make predict FILE=path/to.wav"; exit 1; fi
	$(ACT)
	export PYTHONPATH=.
	$(PY) predict.py \
	  --wav "$(FILE)" \
	  --data_root ../data/GTZAN \
	  --checkpoint artifacts/best_model.pt \
	  --model smallcnn \
	  --topk 5 \
	  --out_dir pred_artifacts

.PHONY: stream
stream:
	$(ACT)
	export PYTHONPATH=.
	$(PY) scripts/stream_infer.py \
	  --data_root ../data/GTZAN \
	  --checkpoint artifacts/best_model.pt \
	  --model smallcnn \
	  --hop_sec 0.25 \
	  $(if $(DEVICE),--device "$(DEVICE)",)

.PHONY: vis
vis:
	$(ACT)
	export PYTHONPATH=.
	$(PY) scripts/vis_dataset.py \
	  --data_root ../data/GTZAN \
	  --split $(if $(SPLIT),$(SPLIT),test) \
	  --checkpoint artifacts/best_model.pt \
	  --model smallcnn \
	  --spec_auto_gain \
	  --sleep 0.5 \
	  --play_audio

.PHONY: demo
demo:
	$(ACT)
	export PYTHONPATH=.
	$(PY) scripts/gen_demo_gif.py \
	  --data_root ../data/GTZAN \
	  --inputs ../data/GTZAN/blues \
	  --checkpoint artifacts/best_model.pt \
	  --model smallcnn \
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
