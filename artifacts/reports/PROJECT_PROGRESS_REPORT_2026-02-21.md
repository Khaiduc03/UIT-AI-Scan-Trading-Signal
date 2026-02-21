# Project Progress Report (Phase 0 -> Phase 9)

Date: 2026-02-21  
Project: AI-TRADING (BTCUSDT 15m, offline)

## 1) Tong quan
- Da hoan thanh Phase 0 -> Phase 8 (pipeline Model 1 + UI export).
- Da implement xong Phase 9 baseline labels cho Model 2 (bar anchor).
- Pipeline hien tai on dinh, artifacts duoc tao day du.

## 2) Tien do theo phase
- Phase 0: Done (repo + config + quickstart)
- Phase 1: Done (download + validate data)
- Phase 2: Done (features core/structure/all)
- Phase 3: Done (StrongMove labels)  
  - Optional 3.2 BreakStructure: Chua lam
- Phase 4: Done (dataset_model1 + time split)
- Phase 5: Done (train Model 1 + metrics)
- Phase 6: Done (scanner report + hotzones + leakage checks)
- Phase 7: Done (scanner tuning summary)
- Phase 8: Done (UI export: hotzones_ui + zoneRisk_points + FE demo folder)
- Phase 9: Done (Model2 label spec + labels artifact + distribution report)

## 3) Da lam duoc gi (ket qua cu the)
### Data/Feature/Label v1
- Raw rows: `73153`
- Features rows:
  - `features_core`: `72853`
  - `features_structure`: `72853`
- StrongMove labels:
  - total rows: `72841`
  - positive rate: `0.726459`

### Dataset split (Model 1)
- train: `50988`
- val: `10926`
- test: `10927`
- Split integrity da pass (time ordered, no overlap).

### Model 1 metrics (test)
- ROC-AUC: `0.6993`
- PR-AUC: `0.8398`
- F1: `0.7632`
- Selected threshold (training policy): `0.50`

### Scanner / Hotzones (config hien tai: threshold=0.75)
- Base rate test: `0.7149`
- Threshold `0.75`:
  - flagged bars: `646`
  - coverage: `0.0591`
  - hit_rate (same-bar): `0.9040`
  - lift_vs_base: `1.2645`
  - hit_within_k_rate: `0.9536`
- Hotzones:
  - total zones: `141`
  - total hot bars: `646`
  - zones/month: `37.1664`

### Phase 9 (Model 2 labels baseline)
- Artifact: `artifacts/processed/labels_model2_bar.parquet`
- Report: `artifacts/reports/model2_label_distribution.json`
- Total labels: `72841`
- Class rates:
  - long: `0.4105`
  - short: `0.4178`
  - neutral: `0.1717`
- Ambiguous both-hit rate: `0.1322`
- Drop stats:
  - dropped_tail_rows: `12`
  - dropped_due_to_atr_rows: `0`
  - dropped_due_to_join_rows: `0`

## 4) Files/artifacts chinh da san sang
- `artifacts/raw/BTCUSDT_15m.csv`
- `artifacts/processed/features_core.parquet`
- `artifacts/processed/features_structure.parquet`
- `artifacts/processed/features_all.parquet`
- `artifacts/processed/labels_strongmove.parquet`
- `artifacts/processed/dataset_model1.parquet`
- `artifacts/processed/train.parquet`
- `artifacts/processed/val.parquet`
- `artifacts/processed/test.parquet`
- `artifacts/models/model1.pkl`
- `artifacts/reports/model1_metrics.json`
- `artifacts/reports/scanner_threshold_report.json`
- `artifacts/reports/hotzones_test.json`
- `artifacts/reports/hotzones_ui.json`
- `artifacts/reports/zoneRisk_points.json`
- `artifacts/reports/leakage_checks.md`
- `artifacts/reports/scanner_tuning_summary.json`
- `artifacts/processed/labels_model2_bar.parquet`
- `artifacts/reports/model2_label_distribution.json`

## 5) Cong viec tiep theo de dat v2
- Phase 10: Build dataset Model 2 (past-only features + time split)
- Phase 11: Zone-sample index cho UI click -> anchor mapping
- Phase 12: Train Model 2 multiclass baseline + metrics
- Phase 13: Context toggle (base vs plus model)
- Phase 14: Export predictions UI-ready cho click-to-predict

## 6) Lenh run nhanh hien tai
```bash
make run-all-phases
make run-model2-labels
make test
```
