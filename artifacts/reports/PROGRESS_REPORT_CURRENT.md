# Progress Report (Current Snapshot)

Date: 2026-02-21  
Repo: /Users/khaimai/workspacing/Personal/lab/AI-TRADING

## 1) Overall status
- Phase 0 -> Phase 8: da trien khai xong flow chinh.
- Phase 3.2 (BreakStructure label): chua implement (optional).
- FE demo folder da co: `fe-demo-model1/`.

## 2) Phase status by roadmap
- Phase 0: Done
- Phase 1: Done
- Phase 2: Done
- Phase 3: Done (StrongMove), Optional 3.2 pending
- Phase 4: Done
- Phase 5: Done
- Phase 6: Done
- Phase 7: Done (tuning summary script + artifact)
- Phase 8: Done (UI export artifacts + FE demo)

## 3) Artifacts check (existence)
- Raw data: `artifacts/raw/BTCUSDT_15m.csv` ✅
- Features: `features_core/structure/all.parquet` ✅
- Labels: `labels_strongmove.parquet` ✅
- Dataset splits: `dataset_model1/train/val/test/splits.json` ✅
- Model: `artifacts/models/model1.pkl` ✅
- Reports core:
  - `data_quality.json` ✅
  - `model1_metrics.json` ✅
  - `scanner_threshold_report.json` ✅
  - `hotzones_test.json` ✅
  - `leakage_checks.md` ✅
- Phase 7:
  - `scanner_tuning_summary.json` ✅
- Phase 8:
  - `hotzones_ui.json` ✅
  - `zoneRisk_points.json` ✅

## 4) Key metrics (current files)
- Data quality rows: `73153`
- Label positive rate (StrongMove): `0.726459`
- Split rows: train `50988`, val `10926`, test `10927`
- Model test PR-AUC: `0.839843`
- Scanner @ threshold 0.80 hit_rate: `0.932749`

## 5) Current scanner config snapshot (configs/config.yaml)
- `hot_threshold = 0.75`
- `min_zone_bars = 2`
- `max_gap_bars = 1`
- `report_thresholds = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]`

## 6) Current hotzone/UI snapshot
- `artifacts/reports/hotzones_ui.json`
  - `total_zones = 141`
  - params: threshold `0.75`, min_zone_bars `2`, max_gap_bars `1`
- `artifacts/reports/zoneRisk_points.json`
  - `rows = 10927`

## 7) Important note about consistency
- `scanner_tuning_summary.json` hien dang la ket qua tuning freeze truoc do (threshold 0.80).
- Config hien tai da doi sang threshold 0.75, nen mot so so lieu moi (zones=141) khac voi tuning summary cu.
- Neu muon dong bo report theo config hien tai, can regenerate:
  1. `make run-scanner-report`
  2. `make run-hotzones`
  3. `make run-export-hotzones-ui`
  4. `make run-export-zonerisk-points`
  5. `make run-scanner-tuning-summary`

## 8) FE demo status
- Folder: `fe-demo-model1/`
- Stack: React + Vite + TS + lightweight-charts
- Features implemented:
  - Candlestick chart
  - Hotzone rectangles overlay
  - ZoneRisk overlay
  - Toggle controls + opacity slider
  - Hover tooltip
  - Click select + sidebar + auto-zoom

