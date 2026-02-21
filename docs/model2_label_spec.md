# Model2 Label Spec (Phase 9)

## Muc tieu
Dinh nghia nhan 3 lop cho Model 2 (trade signal):
- `long`
- `short`
- `neutral`

Phase 9 chi tao labels artifact + distribution report, chua train model.

## Anchor va no-leakage
- Anchor mac dinh: `label2.anchor = "bar"`.
- Moi sample dat tai 1 bar `t`.
- Label su dung future window `[t+1 .. t+K]`.
- Khong duoc dung future data de tao features tai `t`.

## Tham so
Lay tu `configs/config.yaml`:
- `label2.horizon_k` (mac dinh `12`)
- `label2.atr_mult` (mac dinh `1.5`)
- `label2.neutral_rule = "on_tie_or_below_threshold"`
- `label2.mode = "magnitude"`

## Cong thuc
Voi moi sample tai `t`:
- `up_move = max(high[t+1..t+K]) - close[t]`
- `down_move = close[t] - min(low[t+1..t+K])`
- `threshold = atr_mult * atr14[t]`

Gan nhan:
- `long` neu `up_move >= threshold` va `up_move > down_move`
- `short` neu `down_move >= threshold` va `down_move > up_move`
- `neutral` cho cac truong hop con lai (bao gom tie/ambiguous)

## Label ID mapping (locked)
- `neutral = 0`
- `long = 1`
- `short = 2`

## Filter rules
- Drop `K` dong cuoi vi khong du future window.
- Drop dong co `atr14 <= 0` hoac `atr14` null.

## Outputs
- Labels parquet:
  - `artifacts/processed/labels_model2_bar.parquet`
  - Columns: `time, atr14, up_move, down_move, threshold, both_hit, is_tie, neutral_reason, signal, label_id`
- Distribution report:
  - `artifacts/reports/model2_label_distribution.json`
  - Gom class counts/rates, dropped reasons, neutral breakdown, ambiguous metrics, config snapshot.
