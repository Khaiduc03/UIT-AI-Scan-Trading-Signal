```md
# AI-First BTC 15m — Roadmap (Model 1: Hot Zone Scanner) — v1

> Scope v1: **CHỈ AI/DATA/TRAIN/TEST OFFLINE**.  
> Không làm BE/FE trong version này.  
> Market: **BTCUSDT**, timeframe: **15m**  
> Label chuẩn: **K=12 nến (~3h)**, **a=2.5 \* ATR14**  
> Output cuối v1: Model 1 trả **zoneRisk = P(StrongMove)** + report test + danh sách hot zones.

---

## Phase 0 — Bootstrap repo & config

### Goal

Dựng project từ số 0, có cấu trúc rõ ràng, config thống nhất.

### Tasks

- [x] 0.1 Create repo + folder structure
- [x] 0.2 Add config file (single source of truth)
- [x] 0.3 Add README quickstart + commands

### Folder structure (required)
```

ai-first-btc/
src/
data/
features/
labels/
models/
evaluation/
utils/
configs/
artifacts/
raw/
processed/
models/
reports/
tests/
README.md
pyproject.toml (or requirements.txt)

```

### Config (required): `configs/config.yaml`
Must include:
- symbol: BTCUSDT
- timeframe: 15m
- K: 12
- a: 2.5
- atr_period: 14
- train/val/test split (by date or ratio, but time-based)

### Deliverables
- `configs/config.yaml`
- Repo structure created
- `README.md` with run commands

### Acceptance criteria
- Running `python -m src` (or a simple entry command) prints config loaded successfully.

---

## Phase 1 — Data ingestion (OHLCV 15m)

### Goal
Tải dữ liệu BTCUSDT 15m và lưu “raw reproducible”.

### Tasks
- [x] 1.1 Download candles BTCUSDT 15m (≥ 6 months, ideally 1–2 years)
- [x] 1.2 Data validation & data quality report

### Requirements
Raw columns:
- `time, open, high, low, close, volume`
Time must be monotonic increasing, no duplicates.

### Deliverables
- `artifacts/raw/BTCUSDT_15m.csv`
- `artifacts/reports/data_quality.json`

### Acceptance criteria
- No duplicated `time`
- No NaN in required columns
- Candle count >= target (document in report)

---

## Phase 2 — Feature engineering (no lookahead)

### Goal
Tạo features tại thời điểm **t** (chỉ dùng quá khứ/hiện tại).

### Tasks
- [x] 2.1 Core features (volatility, returns, volume)
- [x] 2.2 Structure features (swings + distance to swings)
- [x] 2.3 Merge features into one table

### 2.1 Core features (minimum)
Per bar:
- `atr14`
- `atr_pct = atr14 / close`
- `range1 = high - low`
- `ret1 = (close/close[-1] - 1)`
- `ret3, ret6`
- `abs_ret1`
- `vol_sma50`
- `vol_ratio = volume / vol_sma50`

### 2.2 Structure features (minimum)
Use swing/pivot logic (Lux-style):
- Detect swing highs/lows with a chosen `swing_size` (start with 5 for 15m)
- Compute:
  - `last_swing_high`
  - `last_swing_low`
  - `dist_high_atr = (last_swing_high - close)/atr14`
  - `dist_low_atr  = (close - last_swing_low)/atr14`
  - `near_structure = min(dist_high_atr, dist_low_atr)`

> Note: pivot confirmation introduces delay = `swing_size`. Must be consistent.

### Deliverables
- `artifacts/processed/features_core.parquet`
- `artifacts/processed/features_structure.parquet`
- `artifacts/processed/features_all.parquet`

### Acceptance criteria
- Features table has same row count as raw minus initial warmup
- No future leakage (features at t must not use any bar > t)

---

## Phase 3 — Labeling (K=12, a=2.5*ATR14)

### Goal
Tạo label “StrongMove” dùng tương lai **t+1..t+12**.

### Tasks
- [x] 3.1 Label StrongMove
- [ ] 3.2 (Optional) Label BreakStructure (Up/Down)

### 3.1 StrongMove label (required)
For each bar t:
- `future_range = max(high[t+1..t+K]) - min(low[t+1..t+K])`
- `StrongMove = 1 if future_range >= a * atr14[t] else 0`

### 3.2 BreakStructure labels (optional)
- `BreakUp = 1` if `max(close_future) > last_swing_high(t)`
- `BreakDown = 1` if `min(close_future) < last_swing_low(t)`
- `BreakStructure = BreakUp OR BreakDown`

### Deliverables
- `artifacts/processed/labels_strongmove.parquet`
- `artifacts/processed/labels_breakstructure.parquet` (optional)

### Acceptance criteria
- Last K rows are dropped (cannot label due to missing future window)
- Label distribution (positive rate) is reported

---

## Phase 4 — Build training dataset + time split

### Goal
Ghép features + labels và chia train/val/test theo thời gian.

### Tasks
- [x] 4.1 Merge features_all + labels_strongmove
- [x] 4.2 Time-based split (no shuffle)
- [x] 4.3 Save split metadata

### Deliverables
- `artifacts/processed/dataset_model1.parquet`
- `artifacts/processed/train.parquet`
- `artifacts/processed/val.parquet`
- `artifacts/processed/test.parquet`
- `artifacts/processed/splits.json`

### Acceptance criteria
- Splits are strictly ordered by time
- No overlap between splits

---

## Phase 5 — Train Model 1 (Hot Zone Scanner)

### Goal
Train classifier to output `zoneRisk = P(StrongMove)`.

### Tasks
- [x] 5.1 Train baseline model (LogReg or XGBoost/LightGBM)
- [x] 5.2 Save model artifact
- [x] 5.3 Evaluate on val/test

### Metrics (required)
- ROC-AUC
- PR-AUC
- Precision/Recall/F1 at chosen threshold(s)
- Confusion matrix

### Deliverables
- `artifacts/models/model1.pkl` (or `.json/.txt` depending on framework)
- `artifacts/reports/model1_metrics.json`

### Acceptance criteria
- Model can load and predict on test
- Metrics file exists and contains all required metrics

---

## Phase 6 — Offline AI test (scanner quality) + hot zone extraction

### Goal
Chứng minh AI scanner hữu dụng trước khi làm BE/FE.

### Tasks
- [x] 6.1 Predict `zoneRisk` for all test bars
- [x] 6.2 Threshold report (hit rate vs threshold)
- [x] 6.3 Group consecutive high-risk bars into zones
- [x] 6.4 Leakage sanity checks

### 6.1 Output series
- Add column `zoneRisk` to test set

### 6.2 Threshold report (required)
For thresholds: 0.6, 0.7, 0.75, 0.8
- hit_rate = P(StrongMove=1 | zoneRisk>=thr)
- coverage = %bars flagged as hot
- hit_within_k_rate (optional, future-window early warning)

### 6.3 Zone grouping (required)
Group consecutive bars where `zoneRisk >= 0.75` into zones:
Output per zone:
- `zone_id`
- `from_time`, `to_time`
- `from_index`, `to_index`
- `max_risk`, `avg_risk`
- `count_hot_bars`
- `count_bars_total`

### 6.4 Leakage checks (required)
- Ensure labels use future only; features use present/past only.
- Document any potential leakage risks.

### Deliverables
- `artifacts/reports/zoneRisk_test.parquet`
- `artifacts/reports/scanner_threshold_report.json`
- `artifacts/reports/hotzones_test.json`
- `artifacts/reports/leakage_checks.md`

### Acceptance criteria
- Report shows meaningful lift vs baseline (document baseline used)
- hotzones json renders reasonable number of zones (not zero, not all bars)

---

## Final Output of v1 (What agent must deliver)
- Raw data file: `artifacts/raw/BTCUSDT_15m.csv`
- Dataset: `artifacts/processed/dataset_model1.parquet`
- Splits: train/val/test parquet + `splits.json`
- Model 1 artifact: `artifacts/models/model1.pkl`
- Reports:
  - `model1_metrics.json`
  - `scanner_threshold_report.json`
  - `hotzones_test.json`
  - `leakage_checks.md`

---

## Notes / Decisions locked for v1
- Timeframe: **15m**
- Horizon: **K=12**
- StrongMove threshold: **a=2.5 * ATR14**
- v1 scope: **Model 1 only (scanner)**, **offline test only**
- BE/FE + click-to-predict + Model 2 will be done in **version sau**

---
```
