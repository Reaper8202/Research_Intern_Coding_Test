BOOTSTRAP_PYTHON ?= python3
PYTHON ?= .venv/bin/python

RAW_DIR ?= data/raw
PROCESSED_DIR ?= data/processed
MODELS_DIR ?= models
REPORTS_DIR ?= reports
LOGS_DIR ?= logs

# Transformer settings: override on the command line, e.g. `make finetune FOLD=2 METHOD=H`.
CONFIG ?= configs/transformer.toml
SMOKE_CONFIG ?= configs/smoke.toml
IMAGE_SIZE ?= 64
FOLD ?= 0
METHOD ?= F
RESUME ?=
RESUME_FLAG = $(if $(RESUME),--resume,)
WANDB_ENTITY ?= derek-wu820-university-of-toronto

.PHONY: setup lint format typecheck test check dataset baseline folds preflight \
	modalities tokenize pipeline-smoke pretrain finetune cv-summary wandb-sync clean

# Exact versions from requirements-lock.txt, then the package itself in editable mode.
setup:
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-lock.txt
	$(PYTHON) -m pip install --no-deps -e .

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

check: lint typecheck test

# Step 1: anchors, leak-free history, and labels.
dataset:
	$(PYTHON) -m ehrforecast.build_dataset \
		--data-dir $(RAW_DIR) \
		--out $(PROCESSED_DIR) \
		--reports $(REPORTS_DIR)/dataset \
		--log-file $(LOGS_DIR)/build_dataset.log

# Step 2: logistic-regression baseline on bag-of-codes features.
baseline:
	$(PYTHON) -m ehrforecast.train_baseline \
		--dataset $(PROCESSED_DIR) \
		--models $(MODELS_DIR)/baseline \
		--reports $(REPORTS_DIR)/baseline \
		--log-file $(LOGS_DIR)/train_baseline.log

# Step 3a: persist train-only folds, then audit data and reproduce one baseline configuration.
folds:
	$(PYTHON) -m ehrforecast.folds --dataset $(PROCESSED_DIR) --log-file $(LOGS_DIR)/folds.log

preflight:
	$(PYTHON) -m ehrforecast.preflight \
		--raw $(RAW_DIR) --dataset $(PROCESSED_DIR) --reports $(REPORTS_DIR) \
		--config configs/p3a.toml --log-file $(LOGS_DIR)/preflight.log

# D34: read genomics and DICOM files once into a checked cache (needs `make dataset` first).
modalities:
	$(PYTHON) -m ehrforecast.modalities \
		--raw $(RAW_DIR) --out $(PROCESSED_DIR) --reports $(REPORTS_DIR)/dataset \
		--image-size $(IMAGE_SIZE) --log-file $(LOGS_DIR)/modalities.log

# Steps 3b-3c: fold tokenizers, patient histories, and retention reports (all five folds).
tokenize:
	$(PYTHON) -m ehrforecast.representation \
		--dataset $(PROCESSED_DIR) --config $(CONFIG) --models $(MODELS_DIR)/transformer \
		--reports $(REPORTS_DIR)/representation --log-file $(LOGS_DIR)/tokenize.log

# G2 execution check on fold 0 with tiny settings and temporary outputs (not a result).
pipeline-smoke:
	$(PYTHON) -m ehrforecast.smoke \
		--dataset $(PROCESSED_DIR) --config $(SMOKE_CONFIG) --work $(MODELS_DIR)/smoke \
		--report $(REPORTS_DIR)/transformer/pipeline_$(basename $(notdir $(SMOKE_CONFIG))).json \
		--log-file $(LOGS_DIR)/pipeline_smoke.log

# Step 4: self-pretraining and supervised methods for one fold. Add RESUME=1 to continue.
pretrain:
	$(PYTHON) -m ehrforecast.pretrain \
		--config $(CONFIG) --dataset $(PROCESSED_DIR) --models $(MODELS_DIR)/transformer \
		--fold $(FOLD) $(RESUME_FLAG) --log-file $(LOGS_DIR)/pretrain_fold$(FOLD).log

finetune:
	$(PYTHON) -m ehrforecast.finetune \
		--config $(CONFIG) --dataset $(PROCESSED_DIR) --models $(MODELS_DIR)/transformer \
		--fold $(FOLD) --method $(METHOD) $(RESUME_FLAG) \
		--log-file $(LOGS_DIR)/finetune_fold$(FOLD)_$(METHOD).log

# Step 4: training-CV results and the section 5.3 method selection (no official val).
# Add REFERENCE=configs/transformer.toml to compare a config (e.g. multimodal) with small-v1.
cv-summary:
	$(PYTHON) -m ehrforecast.cv_summary \
		--config $(CONFIG) --dataset $(PROCESSED_DIR) --models $(MODELS_DIR)/transformer \
		--reports $(REPORTS_DIR)/transformer $(if $(REFERENCE),--reference-config $(REFERENCE),) \
		--log-file $(LOGS_DIR)/cv_summary.log

# Upload finished offline wandb runs (runs still training are skipped; safe to repeat).
wandb-sync:
	$(PYTHON) -m ehrforecast.wandb_sync --models $(MODELS_DIR)/transformer \
		--entity $(WANDB_ENTITY) --log-file $(LOGS_DIR)/wandb_sync.log

# Remove generated data, models, and logs (keeps data/raw and the committed reports).
clean:
	rm -rf $(PROCESSED_DIR) $(MODELS_DIR) $(LOGS_DIR)
	rm -rf .mypy_cache .pytest_cache .ruff_cache
