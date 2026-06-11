# LPR pipeline. Each stage reads/writes manifests under data/ — see README.md.
#
#   make help            show targets
#   make test            full verification suite (the only target you need first)
#   make data DATASETS="openalpr ccpd"   download + convert datasets
#   make prep            dedup + group-aware splits + leakage purge
#   make train EPOCHS=20 train the plate head (frozen trunk)
#   make export          ONNX with parity check
#   make engine          TensorRT engine via trtexec (fp16)

# python from the lpr conda env if present, else whatever is on PATH
PY ?= $(shell test -x $(HOME)/miniconda3/envs/lpr/bin/python && echo $(HOME)/miniconda3/envs/lpr/bin/python || echo python3)

VARIANT      ?= s
DATASETS     ?= openalpr
EPOCHS       ?= 20
BATCH        ?= 32
LR           ?= 5e-3
RADIUS       ?= 8
SEED         ?= 0
EXPORT_BATCH ?= 1
ONNX         ?= artifacts/yolov10$(VARIANT)_plate.onnx
ENGINE       ?= artifacts/yolov10$(VARIANT)_plate_fp16.engine

.PHONY: help env weights data data-all prep train test export engine clean

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

env: ## install optional deps (downloaders + onnx); torch/ultralytics assumed present
	$(PY) -m pip install gdown onnx onnxruntime
	@echo "for specific datasets: pip install roboflow (rxg4e/lhqow), fiftyone (open images), kaggle (andrewmvd)"

weights: weights/yolov10$(VARIANT).pt ## fetch official COCO checkpoint

weights/yolov10%.pt:
	mkdir -p weights
	curl -L --fail -o $@ https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov10$*.pt

data: ## download + convert DATASETS (default: openalpr). e.g. make data DATASETS="ccpd crpd"
	$(PY) scripts/build_datasets.py $(DATASETS)

data-all: ## everything in the registry (CCPD ~13GB, CRPD ~19GB; needs API keys for some)
	$(PY) scripts/build_datasets.py all

prep: ## pHash dedup + splits + leakage purge -> data/corpus.parquet
	$(PY) scripts/dedup_and_split.py --radius $(RADIUS) --seed $(SEED)

train: weights ## train plate head (frozen trunk) -> artifacts/plate_head.pt
	$(PY) scripts/train_plate.py --variant $(VARIANT) --weights weights/yolov10$(VARIANT).pt \
	    --epochs $(EPOCHS) --batch-size $(BATCH) --lr $(LR)

test: weights ## full verification suite (parity, surgery, parsers, dedup/split, augment, training)
	$(PY) scripts/verify_yolov10_parity.py $(VARIANT)
	$(PY) scripts/verify_plate_surgery.py $(VARIANT)
	$(PY) scripts/test_parsers.py
	$(PY) scripts/test_dedup_split.py
	$(PY) scripts/test_augment.py
	$(PY) scripts/test_train_smoke.py

export: weights ## export ONNX (incl. plate head if artifacts/plate_head.pt exists) + parity check
	$(PY) scripts/export_onnx.py --variant $(VARIANT) --weights weights/yolov10$(VARIANT).pt --batch $(EXPORT_BATCH)

engine: ## build TensorRT fp16 engine from the ONNX (needs trtexec on PATH)
	@command -v trtexec >/dev/null || { echo "trtexec not found — install TensorRT"; exit 1; }
	trtexec --onnx=$(ONNX) --saveEngine=$(ENGINE) --fp16
	@echo "benchmark numbers are in the trtexec output above; engine: $(ENGINE)"

clean: ## remove generated artifacts (keeps downloaded raw data)
	rm -rf artifacts data/processed data/corpus.parquet
