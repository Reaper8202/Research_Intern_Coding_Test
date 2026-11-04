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
WANDB_MODE ?= online
DEVICE ?= cpu
MANIFEST ?= $(REPORTS_DIR)/final/manifest.json

.PHONY: setup lint format typecheck test check dataset baseline folds preflight \
	modalities tokenize pipeline-smoke pretrain finetune cv-summary experiments \
	experiments-summary lightgbm lightgbm-clean stacking calibration lightgbm-final freeze \
	val-score predict submission package submission-record wandb-sync clean

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

# E11: LightGBM comparison baselines on the training folds (CPU; about an hour).
lightgbm:
	$(PYTHON) -m ehrforecast.lightgbm_cv --dataset $(PROCESSED_DIR) --models $(MODELS_DIR) \
		--reports $(REPORTS_DIR)/lightgbm --compare-config configs/multimodal.toml \
		--log-file $(LOGS_DIR)/lightgbm.log

# D36: the same LightGBM baselines without the simulator clock codes DALY, QALY, and QOLS. This is
# the comparison setting the round-3 stack, the calibration, and the final boosters use.
lightgbm-clean:
	$(PYTHON) -m ehrforecast.lightgbm_cv --dataset $(PROCESSED_DIR) --models $(MODELS_DIR) \
		--reports $(REPORTS_DIR)/lightgbm_clean --output-name lightgbm_clean \
		--exclude-observations DALY QALY QOLS --log-file $(LOGS_DIR)/lightgbm_clean.log

# Tuning round from configs/budget.toml (resumable), then the experiment register.
experiments:
	$(PYTHON) -m ehrforecast.experiments run --budget configs/budget.toml \
		--dataset $(PROCESSED_DIR) --models $(MODELS_DIR)/transformer \
		--logs $(LOGS_DIR)/experiments --log-file $(LOGS_DIR)/experiments.log

experiments-summary:
	$(PYTHON) -m ehrforecast.experiments summarize --budget configs/budget.toml \
		--models $(MODELS_DIR)/transformer --reports $(REPORTS_DIR)/transformer \
		--log-file $(LOGS_DIR)/experiments.log

# Round 3 (configs/budget_v3.toml): CV-only stacking and calibration of the C1 seed ensemble.
C1_CONFIGS = configs/experiments/v2_c1.toml configs/experiments/v2_c1_seed2.toml \
	configs/experiments/v2_c1_seed3.toml

stacking:
	$(PYTHON) -m ehrforecast.stacking --dataset $(PROCESSED_DIR) --models $(MODELS_DIR) \
		--transformer $(C1_CONFIGS) --lightgbm lightgbm_clean \
		--reports $(REPORTS_DIR)/stacking/clean --log-file $(LOGS_DIR)/stacking.log

calibration:
	$(PYTHON) -m ehrforecast.calibration --dataset $(PROCESSED_DIR) --models $(MODELS_DIR) \
		--oof $(REPORTS_DIR)/stacking/clean/oof_stack_logit_mean.parquet \
		--reports $(REPORTS_DIR)/calibration/stack --log-file $(LOGS_DIR)/calibration.log

# Final LightGBM base models: fit on all training patients, save boosters, write val and test
# probabilities (own process: no PyTorch). Frozen with the transformer members by `make freeze`.
lightgbm-final:
	$(PYTHON) -m ehrforecast.lightgbm_final fit --dataset $(PROCESSED_DIR) \
		--settings $(REPORTS_DIR)/lightgbm_clean/cv_summary.json --feature-sets counts counts_values \
		--models $(MODELS_DIR)/lightgbm_final --reports $(REPORTS_DIR)/lightgbm_final \
		--log-file $(LOGS_DIR)/lightgbm_final.log

# P4e: record every artifact hash of the submitted ensemble (configs/final.toml).
freeze:
	$(PYTHON) -m ehrforecast.val_score freeze --config configs/final.toml \
		--dataset $(PROCESSED_DIR) --models $(MODELS_DIR) --manifest $(MANIFEST) \
		--log-file $(LOGS_DIR)/freeze.log

# P4e: score official val once from the frozen manifest; refuses a changed tree or artifact.
val-score:
	$(PYTHON) -m ehrforecast.val_score score --config configs/final.toml \
		--dataset $(PROCESSED_DIR) --models $(MODELS_DIR) --manifest $(MANIFEST) \
		--reports $(REPORTS_DIR)/val --device $(DEVICE) --wandb $(WANDB_MODE) \
		--wandb-entity $(WANDB_ENTITY) --log-file $(LOGS_DIR)/val_score.log

# Step 5: test-split transformer probabilities from the frozen members (reads no labels), then
# the submitted file: the stack of configs/final.toml over the transformer and LightGBM files.
predict:
	$(PYTHON) -m ehrforecast.predict --dataset $(PROCESSED_DIR) --split test \
		--run-dir $$($(PYTHON) -c 'import json; print(" ".join("$(MODELS_DIR)/" + m["run_dir"] for m in json.load(open("$(MANIFEST)"))["members"]))') \
		--device $(DEVICE) --out $(REPORTS_DIR)/final/test_predictions_transformer.csv \
		--log-file $(LOGS_DIR)/predict.log

submission:
	$(PYTHON) -m ehrforecast.submission --config configs/final.toml --dataset $(PROCESSED_DIR) \
		--transformer $(REPORTS_DIR)/final/test_predictions_transformer.csv \
		--lightgbm-dir $(REPORTS_DIR)/lightgbm_final --split test --out predictions.csv \
		--log-file $(LOGS_DIR)/submission.log

# Public package for the Hugging Face Hub (a fresh directory), then the clean-reload check.
package:
	$(PYTHON) -c "import shutil; shutil.rmtree('$(MODELS_DIR)/hf', ignore_errors=True)"
	$(PYTHON) -m ehrforecast.hf_package build --final configs/final.toml --manifest $(MANIFEST) \
		--models $(MODELS_DIR) --dataset $(PROCESSED_DIR) --out $(MODELS_DIR)/hf \
		--configs configs/final.toml configs/v2_base.toml $(C1_CONFIGS) \
		--repo Reaper8202/ehr-forecast --val-summary $(REPORTS_DIR)/val/val_summary.json
	cp docs/model_card.md $(MODELS_DIR)/hf/README.md
	$(PYTHON) -m ehrforecast.hf_package verify --package $(MODELS_DIR)/hf \
		--dataset $(PROCESSED_DIR) --predictions predictions.csv --device $(DEVICE) \
		--out $(REPORTS_DIR)/final/package_verify_cpu.json

# Hashes of the files that leave the repository, with the commit (run after the final PDF).
submission-record:
	$(PYTHON) scripts/submission_record.py --out $(REPORTS_DIR)/final/submission_record.json

# Upload finished offline wandb runs (runs still training are skipped; safe to repeat).
wandb-sync:
	$(PYTHON) -m ehrforecast.wandb_sync --models $(MODELS_DIR)/transformer \
		--entity $(WANDB_ENTITY) --log-file $(LOGS_DIR)/wandb_sync.log

# Remove generated data, models, and logs (keeps data/raw and the committed reports).
clean:
	rm -rf $(PROCESSED_DIR) $(MODELS_DIR) $(LOGS_DIR)
	rm -rf .mypy_cache .pytest_cache .ruff_cache
