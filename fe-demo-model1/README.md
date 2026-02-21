# FE Demo Model 1 — Hot Zones + ZoneRisk Overlay

Single-page frontend demo for Model 1 scanner outputs (static JSON only, no backend).

## Stack
- React + Vite + TypeScript
- TradingView Lightweight Charts (`lightweight-charts`)

## Install & Run
```bash
cd /Users/khaimai/workspacing/Personal/lab/AI-TRADING/fe-demo-model1
npm install
npm run dev
```

## Data Inputs (`public/data/`)
Required files:
- `hotzones_ui.json`
- `zoneRisk_points.json`
- `candles_test.json`

Current dataset in this repo is already prepared in:
- `public/data/hotzones_ui.json`
- `public/data/zoneRisk_points.json`
- `public/data/candles_test.json`

## UI Behavior
- Candlestick chart from `candles_test.json`
- Hotzone rectangles from `hotzones_ui.json`
- Optional ZoneRisk overlay line from `zoneRisk_points.json`
- Controls:
  - Toggle Hotzones
  - Toggle ZoneRisk
  - Opacity slider (`0.10 -> 0.60`)
- Interactions:
  - Hover zone => tooltip (`zone_id`, `max_risk`, `avg_risk`, `from_time`, `to_time`)
  - Click zone => select + highlight + right sidebar details + auto-zoom to zone range

## FE Rendering Contract
- Parse ISO UTC to unix seconds in code (`parseIsoToUnixSeconds`).
- Rectangle bounds:
  - `x = [from_time, to_time]`
  - `y = [bottom_price, top_price]`
- Zone count depends on frozen scanner params; FE only visualizes provided artifacts.

## Notes
- This FE demo is visualization-only and does not require backend APIs.
- If source artifacts are regenerated in root repo, copy updated JSON files to `public/data/`.
