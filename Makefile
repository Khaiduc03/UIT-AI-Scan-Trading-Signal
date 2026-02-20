PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup dev lock check lint format test run-download run-validate run-features run-labels run-dataset run-dataset-checks clean

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
	$(PYTHON) src/data/download_data.py

run-validate:
	$(PYTHON) src/data/validate_data.py

run-features:
	$(PYTHON) src/features/build_features.py

run-labels:
	$(PYTHON) src/labels/build_labels.py

run-dataset:
	$(PYTHON) src/models/build_dataset.py

run-dataset-checks:
	$(PYTHON) src/evaluation/check_dataset_splits.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
