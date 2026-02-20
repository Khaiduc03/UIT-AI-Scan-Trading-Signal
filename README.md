# AI-TRADING

AI-powered Bitcoin trading system.

## Environment setup (.venv)

```bash
cd /Users/khaimai/workspacing/Personal/lab/AI-TRADING
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Reproducible install on another device

```bash
cd /Users/khaimai/workspacing/Personal/lab/AI-TRADING
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run feature engineering (Phase 2)

```bash
source .venv/bin/activate
python src/features/build_features.py
```

Outputs:
- `artifacts/processed/features_core.parquet`
- `artifacts/processed/features_structure.parquet`

## Run labeling (Phase 3)

```bash
source .venv/bin/activate
python src/labels/build_labels.py
```

Outputs:
- `artifacts/processed/labels_strongmove.parquet`
- `artifacts/reports/labels_strongmove_report.json`

## Run dataset build + time split (Phase 4)

```bash
source .venv/bin/activate
python src/models/build_dataset.py
```

Outputs:
- `artifacts/processed/dataset_model1.parquet`
- `artifacts/processed/train.parquet`
- `artifacts/processed/val.parquet`
- `artifacts/processed/test.parquet`
- `artifacts/processed/splits.json`

## Run model training (Phase 5)

```bash
source .venv/bin/activate
python src/models/train_model1.py
```

Outputs:
- `artifacts/models/model1.pkl`
- `artifacts/reports/model1_metrics.json`

## Project Rules

- Rules document: `PROJECT_RULES.md`
- Main conventions:
  - Use `.venv` only.
  - `pyproject.toml` is source-of-truth; `requirements.txt` is lock snapshot.
  - Run quality gate before commit: `make check`.

## Make Commands

```bash
make setup          # create .venv + install runtime deps
make dev            # install runtime + dev deps
make lock           # refresh requirements.txt from current .venv
make lint           # run ruff checks
make test           # run pytest
make check          # lint + test
make run-download   # download raw OHLCV
make run-validate   # validate raw data
make run-features   # build phase-2 features
make run-labels     # build phase-3 StrongMove labels
make run-dataset    # build phase-4 dataset + time split
make run-dataset-checks # check split consistency, class balance, drift
make run-train-model1 # train phase-5 baseline model + metrics report
```

## Command meanings

- `make run-download`: tải dữ liệu BTCUSDT 15m vào `artifacts/raw/`.
- `make run-validate`: kiểm tra chất lượng dữ liệu raw và ghi report.
- `make run-features`: tạo features Phase 2 (`features_core`, `features_structure`).
- `make run-labels`: tạo label StrongMove + report phân phối nhãn.
- `make run-dataset`: ghép features + labels và chia train/val/test theo thời gian.
- `make run-dataset-checks`: kiểm tra integrity split, class balance, drift.
- `make run-train-model1`: train baseline logreg và ghi model/metrics của Phase 5.

## Project Structure

- `src/`: Source code
  - `data/`: Data loading/processing
  - `features/`: Feature engineering
  - `labels/`: Labeling logic
  - `models/`: Model architecture/training
  - `evaluation/`: Backtesting and metrics
  - `utils/`: Common utilities
- `notebooks/`: Research and exploration
- `configs/`: Configuration files
- `artifacts/`: Project artifacts
  - `raw/`: Raw data
  - `processed/`: Processed data
  - `models/`: Saved models
  - `reports/`: Evaluation reports
- `tests/`: Unit and integration tests
