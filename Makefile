BOOTSTRAP_PYTHON ?= python3
PYTHON ?= .venv/bin/python

RAW_DIR ?= data/raw
PROCESSED_DIR ?= data/processed
MODELS_DIR ?= models
REPORTS_DIR ?= reports
LOGS_DIR ?= logs

.PHONY: setup lint format typecheck test check dataset baseline clean

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

# Remove generated data, models, and logs (keeps data/raw and the committed reports).
clean:
	rm -rf $(PROCESSED_DIR) $(MODELS_DIR) $(LOGS_DIR)
	rm -rf .mypy_cache .pytest_cache .ruff_cache
