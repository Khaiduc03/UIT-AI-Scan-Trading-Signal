# Phase 8 Report — UI Demo Export (v1)

Date: 2026-02-21  
Project: AI-TRADING (BTCUSDT 15m, offline v1)

## Muc tieu
Xuat du lieu san sang cho FE de ve:
- Hotzone rectangles (`from_time/to_time` + `bottom_price/top_price`)
- ZoneRisk overlay series (`time`, `zoneRisk`)

## Inputs da dung
- `artifacts/raw/BTCUSDT_15m.csv`
- `artifacts/reports/zoneRisk_test.parquet`
- `artifacts/reports/hotzones_test.json`
- `configs/config.yaml`

## Outputs Phase 8
- `artifacts/reports/hotzones_ui.json`
- `artifacts/reports/zoneRisk_points.json`

## Ket qua thuc te
- `timeframe = 15m`
- `test_start_time = 2025-10-10T01:30:00Z`
- `test_end_time = 2026-01-31T21:00:00Z`
- `test_rows = 10927`
- `total_zones = 65`
- Frozen scanner params:
- `hot_threshold = 0.80`
- `min_zone_bars = 2`
- `max_gap_bars = 1`
- zoneRisk points:
- `rows = 10927`

## Quality snapshot (tham chieu Phase 7)
- `flagged_bars@0.80 = 342`
- `coverage@0.80 = 0.0313`
- `hit_rate@0.80 = 0.9327`
- `lift_vs_base@0.80 = 1.3047`
- `hit_within_k_rate@0.80 = 0.9503`
- `zones_per_month = 17.1334` (tren test window)

## FE rendering contract
- Rectangle:
- `x = [from_time, to_time]`
- `y = [bottom_price, top_price]`
- label goi y: `zone_id`, `max_risk`
- Overlay:
- dung `zoneRisk_points.json` (`time`, `zoneRisk`) de ve line/area.

## Vi du zone dau tien
- `zone_id = 1`
- `from_time = 2025-10-10T13:30:00Z`
- `to_time = 2025-10-10T13:45:00Z`
- `bottom_price = 121384.6`
- `top_price = 122550.0`

## Lenh reproduce
```bash
cd /Users/khaimai/workspacing/Personal/lab/AI-TRADING
source .venv/bin/activate
make run-export-hotzones-ui
make run-export-zonerisk-points
# or
make run-phase8-ui
```

## Validation
- `make lint` passed
- `make test` passed (`33 passed`)
- `hotzones_ui.total_zones == len(hotzones_ui.zones)` true
- `zoneRisk_points.rows == zoneRisk_test rows` true
- `zoneRisk` values nam trong `[0,1]`
