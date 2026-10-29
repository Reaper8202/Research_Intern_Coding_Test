PYTHON ?= .venv/bin/python
DATA_DIR ?= data
ARTIFACTS ?= artifacts

.PHONY: setup lint typecheck test check dataset baseline

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m pytest

check: lint typecheck test

# Step 1: anchors, leak-free history, and labels.
dataset:
	$(PYTHON) -m ehrforecast.build_dataset --data-dir $(DATA_DIR) --out $(ARTIFACTS)/dataset

# Step 2: logistic-regression baseline on bag-of-codes features.
baseline:
	$(PYTHON) -m ehrforecast.train_baseline --dataset $(ARTIFACTS)/dataset --out $(ARTIFACTS)/baseline
