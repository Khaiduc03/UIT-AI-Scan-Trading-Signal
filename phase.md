# AI-TRADING — Roadmap (BTCUSDT 15m) — v1 DONE + v2 (Model 2) NEXT

> Market: **BTCUSDT**, timeframe: **15m**  
> v1 label: **StrongMove** with **K=12 (~3h)**, **a=2.5 \* ATR14**  
> v1 output: **zoneRisk = P(StrongMove)** + hotzones + UI demo export  
> v2 goal: **User selects a zone → Model 2 returns Signal + Probabilities (+ optional Entry/SL/TP rule-based)** (offline first)

---

## Global Current Settings

### v1 (current)

- `label.horizon_k = 12`
- `label.strongmove_atr_mult = 2.5`
- `features.atr_period = 14`
- `scanner.hot_threshold = 0.75`
- `scanner.min_zone_bars = 2`
- `scanner.max_gap_bars = 1`
- `scanner.report_thresholds = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]`

### v2 (new required UX behavior)

- In UI “Predict” panel:
  - `use_model1_context` (boolean)
  - **default = false**
- If `use_model1_context=false` → Model 2 uses ONLY its own candle-based/context features.
- If `use_model1_context=true` → Model 2 additionally consumes Model 1 context (zoneRisk + zone stats) as extra features.

---

# v1 — Model 1 Hot Zone Scanner (Phase 0 → Phase 8) ✅ DONE

(unchanged)

---

# v2 — Model 2 Trade Signal (Offline First)

## Phase 9 — Define Model 2 task + label set (bar-level baseline)

### Goal

Chốt rõ: Model 2 dự đoán gì khi user bấm **Predict**.

### Output contract (locked)

- `signal`: `Long | Short | Neutral`
- `probabilities`: `p_long, p_short, p_neutral`
- `confidence`: `max(probabilities)`
- Optional (baseline rule-based first, NOT ML yet):
  - `entry`, `stop_loss`, `take_profit`

### Label baseline (locked: Option A)

**3-class direction by future move** (bar anchored)

For each sample at anchor time `t` (each bar):

- Lookahead window: `t+1 .. t+K`
- Compute:
  - `up_move = max(high_future) - close[t]`
  - `down_move = close[t] - min(low_future)`
- Define:
  - `Long` if `up_move >= b * ATR14(t)` and `up_move > down_move`
  - `Short` if `down_move >= b * ATR14(t)` and `down_move > up_move`
  - else `Neutral`

Params:

- `K`: default `12` (match v1), optional try `24` later
- `b`: default `1.5` (tune later)
- `neutral_rule`: `on_tie_or_below_threshold`

### Deliverables

- `docs/model2_label_spec.md`
- Config add sections:
  - `label2: { anchor, horizon_k, atr_mult, neutral_rule }`
  - `model2: { use_model1_context_default: false }`
- `artifacts/processed/labels_model2_bar.parquet`
- `artifacts/reports/model2_label_distribution.json`

### Acceptance

- Labels computed without leakage into features
- Class balance report exists

---

## Phase 10 — Build Model 2 dataset (anchor-based, candle-only baseline)

### Goal

Tạo **bar-level dataset** cho Model 2 (không phụ thuộc Model 1) để model có thể chạy ở mọi thời điểm.

### Decision (locked)

- Each training row corresponds to a single anchor time `t0`
- Features are computed from **past window only**: `[t0-L+1 .. t0]`
- Label uses future window only: `[t0+1 .. t0+K2]`

Recommended defaults:

- `L = 96` bars (24h) or `L = 48` (12h) — tune later

### Outputs

- `artifacts/processed/dataset_model2.parquet`
- `artifacts/reports/model2_dataset_report.json`

### Acceptance

- No future bars used in features
- Time monotonic, unique per `t0`

---

## Phase 11 — Optional: Build “zone-sample” index (for UI selection)

### Goal

UI bấm vào 1 zone → map sang anchor time `t0` để gọi Model 2 predict.

### How

- For each zone, define:
  - `t0 = zone.to_time` (zone end)
  - Keep: `zone_id, from_time, to_time, from_index, to_index, top/bottom`
- Save a UI-friendly mapping file:
  - `artifacts/reports/model2_zone_index_test.json`

### Acceptance

- Every zone maps to exactly 1 anchor time `t0` in test timeline

---

## Phase 12 — Train Model 2 (multiclass baseline)

### Goal

Train baseline model for `Long/Short/Neutral`.

### Baseline model (locked)

- Multinomial Logistic Regression (fast, explainable)

### Outputs

- `artifacts/models/model2.pkl`
- `artifacts/reports/model2_metrics.json`

### Metrics (required)

- Macro F1
- Per-class precision/recall/F1
- Confusion matrix
- Calibration bins (optional)

### Acceptance

- Model loads and predicts on test
- Metrics complete

---

## Phase 13 — Model 2 “Context Toggle” upgrade (use_model1_context = optional)

### Goal

Cho phép Model 2 nhận thêm context từ Model 1 khi user bật option.

### Design (locked)

- Keep two feature sets:
  1. **Base features** (candle-only) — always available
  2. **Context features** (from Model 1) — optional

Context feature candidates:

- `zoneRisk(t0)` (from `zoneRisk_test.parquet` during offline eval)
- `zoneRisk_slope_recent` (e.g. slope last 8 bars ending at t0)
- `hot_threshold_used` (constant, optional)
- If zone selected: `zone_len_bars`, `max_risk`, `avg_risk` (from `hotzones_ui.json`)

### Config (locked)

- `model2.use_model1_context_default: false`
- `model2.allow_model1_context: true`

### Training strategy (recommended)

- Train **two artifacts**:
  - `model2_base.pkl` (no context)
  - `model2_plus.pkl` (with context)
- Report comparison:
  - `artifacts/reports/model2_ablation_report.json`

### Acceptance

- Both models can predict
- Plus model is only used when toggle=true

---

## Phase 14 — Offline “Click Zone → Predict” export (UI-ready)

### Goal

Xuất kết quả để FE demo hiển thị panel khi click zone.

### Output schema (locked)

`artifacts/reports/model2_predictions_test.json`:

- `zone_id`
- `t0_time`
- `use_model1_context` (true/false)
- `signal`
- `probabilities`
- `confidence`
- Optional rule-based:
  - `entry`, `stop_loss`, `take_profit`
- `notes` / `explanations` (top features)

### Acceptance

- Predict works for any `zone_id` in test
- Output is static-json ready for FE

---

## v2 Deliverables (minimum)

- `dataset_model2.parquet` + splits
- `model2_base.pkl` + `model2_metrics.json`
- (optional) `model2_plus.pkl` + `model2_ablation_report.json`
- `model2_predictions_test.json` (UI-ready)
- `model2_zone_index_test.json` (mapping zones → t0)

---
