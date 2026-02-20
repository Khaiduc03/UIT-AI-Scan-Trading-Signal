PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup dev lock check lint format test run-download run-validate run-features run-labels run-dataset run-dataset-checks run-train-model1 clean

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e .

dev: setup
	$(PIP) install -e ".[dev]"

lock:
	$(PIP) freeze | rg -v "^-e " > requirements.txt

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

test:
	$(PYTHON) -m pytest -q tests

check: lint test

run-download:
	@echo "[run-download] Download BTC OHLCV raw data from exchange into artifacts/raw."
	$(PYTHON) src/data/download_data.py

run-validate:
	@echo "[run-validate] Validate raw OHLCV quality (columns/sort/NaN/duplicates)."
	$(PYTHON) src/data/validate_data.py

run-features:
	@echo "[run-features] Build Phase 2 features_core and features_structure."
	$(PYTHON) src/features/build_features.py

run-labels:
	@echo "[run-labels] Build Phase 3 StrongMove labels and distribution report."
	$(PYTHON) src/labels/build_labels.py

run-dataset:
	@echo "[run-dataset] Build Phase 4 dataset_model1 and time-based train/val/test splits."
	$(PYTHON) src/models/build_dataset.py

run-dataset-checks:
	@echo "[run-dataset-checks] Check split integrity, class balance, and train-vs-test drift."
	$(PYTHON) src/evaluation/check_dataset_splits.py

run-train-model1:
	@echo "[run-train-model1] Train Phase 5 baseline model and export model + metrics artifacts."
	$(PYTHON) src/models/train_model1.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
