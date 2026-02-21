# Bao Cao Tong Hop Phase 1-7 (AI-TRADING v1)

Ngay cap nhat: 2026-02-21  
Project: AI-First BTCUSDT 15m - Model 1 Hot Zone Scanner

## Tong quan trang thai
- Phase 1 den Phase 7 da hoan thanh pipeline offline.
- Tat ca test hien tai pass: `31 passed`.
- Leakage checks: `PASS`.

## Phase 1 - Data ingestion
- Raw file: `artifacts/raw/BTCUSDT_15m.csv`
- Data quality report: `artifacts/reports/data_quality.json`
- Ket qua:
- `total_rows = 73153`
- `is_sorted = true`
- `has_duplicates = false`
- `has_nans = false`
- `is_valid = true`

## Phase 2 - Feature engineering
- Outputs:
- `artifacts/processed/features_core.parquet`
- `artifacts/processed/features_structure.parquet`
- `artifacts/processed/features_all.parquet`
- Ket qua:
- `features_core rows = 72853`
- `features_structure rows = 72853`
- `features_all rows = 72853`
- Time range features: `2024-01-04T03:00:00Z -> 2026-02-01T00:00:00Z`

## Phase 3 - Labeling (StrongMove)
- Output:
- `artifacts/processed/labels_strongmove.parquet`
- `artifacts/reports/labels_strongmove_report.json`
- Ket qua:
- `total_rows = 72841`
- `positive_count = 52916`
- `negative_count = 19925`
- `positive_rate = 0.726459`
- `horizon_k = 12`
- `strongmove_atr_mult = 2.5`

## Phase 4 - Dataset + time split
- Outputs:
- `artifacts/processed/dataset_model1.parquet`
- `artifacts/processed/train.parquet`
- `artifacts/processed/val.parquet`
- `artifacts/processed/test.parquet`
- `artifacts/processed/splits.json`
- Ket qua split:
- Dataset rows: `72841`
- Train: `50988` (0.69999)
- Val: `10926` (0.14999)
- Test: `10927` (0.15001)
- Integrity:
- `is_time_sorted = true`
- `has_overlap = false`
- `is_disjoint_complete_partition = true`

## Phase 5 - Train Model 1
- Outputs:
- `artifacts/models/model1.pkl`
- `artifacts/reports/model1_metrics.json`
- Cau hinh model:
- `type = logreg`
- `feature_count = 14`
- `selected_threshold = 0.50` (validation policy: `f_beta`, beta=0.5)
- Test metrics:
- `ROC-AUC = 0.6993`
- `PR-AUC = 0.8398`
- `Precision = 0.8114`
- `Recall = 0.7204`
- `F1 = 0.7632`
- Confusion matrix (test):
- `TN=1807, FP=1308, FN=2184, TP=5628`

## Phase 6 - Offline scanner + hotzones + leakage checks
- Outputs:
- `artifacts/reports/zoneRisk_test.parquet`
- `artifacts/reports/scanner_threshold_report.json`
- `artifacts/reports/hotzones_test.json`
- `artifacts/reports/leakage_checks.md`
- Scanner quality (threshold 0.80):
- `flagged_bars = 342`
- `coverage = 0.0313`
- `hit_rate = 0.9327`
- `lift_vs_base = 1.3047`
- `hit_within_k_rate = 0.9503`
- Hotzones:
- `total_zones = 65`
- `total_hot_bars = 342`
- Leakage report:
- `Overall status: PASS`

## Phase 7 - Scanner tuning freeze (v1)
- Muc tieu: freeze tham so scanner + tao summary report cho GitHub.
- Frozen params trong config:
- `hot_threshold = 0.80`
- `min_zone_bars = 2`
- `max_gap_bars = 1`
- `report_thresholds = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]`
- New output:
- `artifacts/reports/scanner_tuning_summary.json`
- Ket qua tuning summary:
- `test_rows = 10927`
- `months_in_test = 3.79375`
- `total_zones = 65`
- `zones_per_month = 17.1334`
- `total_hot_bars = 342`
- `coverage_hot_bars = 0.0313`

## Deliverables chinh de bao cao
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
