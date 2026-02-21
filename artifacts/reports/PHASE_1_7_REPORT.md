# Phase 1-7 Execution Report (GitHub Ready)

Date: 2026-02-21  
Project: AI-TRADING (AI-First BTCUSDT 15m, Model 1 Hot Zone Scanner, offline v1)

## 1) Executive Summary
- Pipeline Phase 1 -> Phase 7 completed end-to-end in offline mode.
- Quality gates passed: `make lint` and `make test` (`31 passed`).
- Leakage checks: `PASS`.
- Data source: Binance spot via CCXT.
- Final scanner freeze for v1:
- `hot_threshold = 0.80`
- `min_zone_bars = 2`
- `max_gap_bars = 1`
- `report_thresholds = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]`

## 2) Config Snapshot (Reproducibility)
Source file: `configs/config.yaml`

```yaml
data:
  symbol: BTCUSDT
  timeframe: 15m
  start_date: 2024-01-01
  end_date: 2026-02-01

features:
  warmup_bars: 300
  swing_size: 5
  atr_period: 14

label:
  horizon_k: 12
  strongmove_atr_mult: 2.5

split:
  method: time_ratio
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

model1:
  type: logreg
  class_weight: balanced
  solver: liblinear
  max_iter: 1000
```

## 3) Environment Snapshot
- Python: `3.10.9`
- Environment: `.venv`
- Dependency sources:
- `pyproject.toml` (primary)
- `requirements.txt` (locked snapshot)

## 4) Phase-by-Phase Results

### Phase 1 - Data ingestion
- Raw data: `artifacts/raw/BTCUSDT_15m.csv`
- Data quality report: `artifacts/reports/data_quality.json`
- Results:
- `total_rows = 73153`
- `raw_time_range = 2024-01-01T00:00:00Z -> 2026-02-01T00:00:00Z`
- `is_sorted = true`
- `has_duplicates = false`
- `has_nans = false`
- `is_valid = true`

### Phase 2 - Feature engineering
- Outputs:
- `artifacts/processed/features_core.parquet`
- `artifacts/processed/features_structure.parquet`
- `artifacts/processed/features_all.parquet`
- Results:
- `features_core rows = 72853`
- `features_structure rows = 72853`
- `features_all rows = 72853`
- Feature time range: `2024-01-04T03:00:00Z -> 2026-02-01T00:00:00Z`

### Phase 3 - Labeling (StrongMove)
- Outputs:
- `artifacts/processed/labels_strongmove.parquet`
- `artifacts/reports/labels_strongmove_report.json`
- Results:
- `total_rows = 72841`
- `positive_count = 52916`
- `negative_count = 19925`
- `positive_rate = 0.726459`
- `horizon_k = 12`
- `strongmove_atr_mult = 2.5`

### Phase 4 - Dataset + time split
- Outputs:
- `artifacts/processed/dataset_model1.parquet`
- `artifacts/processed/train.parquet`
- `artifacts/processed/val.parquet`
- `artifacts/processed/test.parquet`
- `artifacts/processed/splits.json`
- Results:
- `dataset rows = 72841`
- `train rows = 50988`
- `val rows = 10926`
- `test rows = 10927`
- Integrity:
- `is_time_sorted = true`
- `has_overlap = false`
- `is_disjoint_complete_partition = true`

### Phase 5 - Model 1 training
- Outputs:
- `artifacts/models/model1.pkl`
- `artifacts/reports/model1_metrics.json`
- Model summary:
- `type = logreg`
- `feature_count = 14`
- `selected_threshold = 0.50` (policy `f_beta`, beta=0.5)
- Feature columns:
- `["atr14","atr_pct","range1","ret1","ret3","ret6","abs_ret1","vol_sma50","vol_ratio","last_swing_high","last_swing_low","dist_high_atr","dist_low_atr","near_structure"]`
- Test metrics:
- `ROC-AUC = 0.6993`
- `PR-AUC = 0.8398`
- `Precision = 0.8114`
- `Recall = 0.7204`
- `F1 = 0.7632`
- Confusion matrix:
- `TN=1807, FP=1308, FN=2184, TP=5628`

### Phase 6 - Scanner report + hotzones + leakage
- Outputs:
- `artifacts/reports/zoneRisk_test.parquet`
- `artifacts/reports/scanner_threshold_report.json`
- `artifacts/reports/hotzones_test.json`
- `artifacts/reports/leakage_checks.md`
- Test range used by scanner:
- `test_start_time = 2025-10-10T01:30:00Z`
- `test_end_time = 2026-01-31T21:00:00Z`
- `test_rows = 10927`
- Baseline:
- `base_rate_test = 0.714926`
- Scanner quality at threshold `0.80`:
- `flagged_bars = 342`
- `coverage = 0.0313`
- `hit_rate = P(StrongMove=1 | zoneRisk >= thr) = 0.9327` (same-bar)
- `precision_at_thr = 0.9327`
- `lift_vs_base = 1.3047`
- `hit_within_k_rate = 0.9503`
- Hotzones at current frozen params:
- `total_zones = 65`
- `total_hot_bars = 342`
- Leakage:
- `Overall status: PASS`

### Phase 7 - Scanner tuning freeze
- New output:
- `artifacts/reports/scanner_tuning_summary.json`
- Final frozen decision for v1:
- Keep `hot_threshold=0.80` as final production-like offline setting.
- Note: `0.85` may reduce zones further, but is not frozen for v1.
- Density summary:
- `months_in_test = 3.79375`
- `total_zones = 65`
- `zones_per_month = 17.1334`
- `total_hot_bars = 342`
- `coverage_hot_bars = 0.0313`
- Density is computed on test window only.

## 5) Reproduce Commands

```bash
cd /Users/khaimai/workspacing/Personal/lab/AI-TRADING
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
# fallback reproducible lock install (if needed):
python -m pip install -r requirements.txt
```

Run full pipeline (Phase 1 -> 6):

```bash
make run-all-phases
```

Run Phase 7 summary (after Phase 6 artifacts exist):

```bash
make run-scanner-tuning-summary
```

Validate quality gates:

```bash
make lint
make test
```

## 6) Final Deliverables
- `artifacts/raw/BTCUSDT_15m.csv`
- `artifacts/processed/dataset_model1.parquet`
- `artifacts/processed/train.parquet`
- `artifacts/processed/val.parquet`
- `artifacts/processed/test.parquet`
- `artifacts/processed/splits.json`
- `artifacts/models/model1.pkl`
- `artifacts/reports/model1_metrics.json`
- `artifacts/reports/scanner_threshold_report.json`
- `artifacts/reports/hotzones_test.json`
- `artifacts/reports/leakage_checks.md`
- `artifacts/reports/scanner_tuning_summary.json`
