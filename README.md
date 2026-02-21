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

## Run Model2 labeling baseline (Phase 9)

```bash
source .venv/bin/activate
python src/labels/build_labels_model2.py
```

Outputs:
- `artifacts/processed/labels_model2_bar.parquet`
- `artifacts/reports/model2_label_distribution.json`

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

## Run scanner offline report (Phase 6 Prompt 1)

```bash
source .venv/bin/activate
python src/evaluation/run_scanner_report.py
```

Outputs:
- `artifacts/reports/zoneRisk_test.parquet`
- `artifacts/reports/scanner_threshold_report.json`

## Run hotzone extraction (Phase 6 Prompt 2)

```bash
source .venv/bin/activate
python src/evaluation/extract_hotzones.py
```

Outputs:
- `artifacts/reports/hotzones_test.json`

## Run scanner tuning freeze summary (Phase 7)

```bash
source .venv/bin/activate
python src/evaluation/write_scanner_tuning_summary.py
```

Outputs:
- `artifacts/reports/scanner_tuning_summary.json`

## Export hotzones for UI rectangles (Phase 8)

```bash
source .venv/bin/activate
python src/evaluation/export_hotzones_ui.py
```

Outputs:
- `artifacts/reports/hotzones_ui.json`

Schema (top-level):
- `timeframe`
- `params` (`hot_threshold`, `min_zone_bars`, `max_gap_bars`)
- `test_range` (`start_time`, `end_time`, `rows`)
- `total_zones`
- `zones`

FE rendering notes:
- Draw zone rectangle with `x=[from_time,to_time]`, `y=[bottom_price,top_price]`.
- Optionally show `zone_id` and `max_risk` as label.

## Export zoneRisk overlay points (Phase 8, optional)

```bash
source .venv/bin/activate
python src/evaluation/export_zoneRisk_points.py
```

Outputs:
- `artifacts/reports/zoneRisk_points.json`

## Run leakage sanity checks (Phase 6 Prompt 3)

```bash
source .venv/bin/activate
python src/evaluation/leakage_checks_phase6.py
```

Outputs:
- `artifacts/reports/leakage_checks.md`

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
make run-model2-labels # build phase-9 Model2 labels + distribution report
make run-dataset    # build phase-4 dataset + time split
make run-dataset-checks # check split consistency, class balance, drift
make run-train-model1 # train phase-5 baseline model + metrics report
make run-scanner-report # phase-6 prompt 1: zoneRisk + threshold report
make run-hotzones # phase-6 prompt 2: hotzone extraction
make run-leakage-checks # phase-6 prompt 3: leakage + artifact sanity checks
make run-scanner-tuning-summary # phase-7: scanner tuning freeze summary
make run-export-hotzones-ui # phase-8: hotzones UI rectangles export
make run-export-zonerisk-points # phase-8: zoneRisk overlay points export
make run-phase8-ui # phase-8 convenience target (run both exports)
make run-all-phases # run full pipeline phase 1 -> phase 6
```

## Command meanings

- `make run-download`: tải dữ liệu BTCUSDT 15m vào `artifacts/raw/`.
  Nếu file raw đã tồn tại, lệnh sẽ hỏi có tải lại hay không (`y/N`).
  Dùng `REDOWNLOAD=1 make run-download` để ép tải lại trong non-interactive mode.
- `make run-validate`: kiểm tra chất lượng dữ liệu raw và ghi report.
- `make run-features`: tạo features Phase 2 (`features_core`, `features_structure`).
- `make run-labels`: tạo label StrongMove + report phân phối nhãn.
- `make run-dataset`: ghép features + labels và chia train/val/test theo thời gian.
- `make run-dataset-checks`: kiểm tra integrity split, class balance, drift.
- `make run-train-model1`: train baseline logreg và ghi model/metrics của Phase 5.
- `make run-scanner-report`: tạo zoneRisk trên test + threshold quality report (>= threshold).
- `make run-hotzones`: trích xuất hot zones từ zoneRisk với `max_gap_bars`.
- `make run-leakage-checks`: sinh `leakage_checks.md` và kiểm tra chéo artifacts Phase 6.
- `make run-scanner-tuning-summary`: tổng hợp báo cáo tuning scanner để freeze cấu hình cho v1.
- `make run-export-hotzones-ui`: export dữ liệu zone rectangle cho FE.
- `make run-export-zonerisk-points`: export chuỗi time + zoneRisk để overlay chart.
- `make run-phase8-ui`: chạy cả 2 export Phase 8.

## Scanner Freeze (v1)

- `hot_threshold=0.75`
- `min_zone_bars=2`
- `max_gap_bars=1`
- `report_thresholds=[0.60, 0.70, 0.75, 0.80, 0.85, 0.90]`

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
