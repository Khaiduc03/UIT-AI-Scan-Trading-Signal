# Project Rules

## 1) Environment
- Always run commands through project venv: `.venv`.
- Python version target: `3.10+`.
- Install:
  - `make setup` for runtime deps.
  - `make dev` for runtime + dev deps.

## 2) Source of truth for dependencies
- `pyproject.toml` is the primary dependency declaration.
- `requirements.txt` is a pinned snapshot for reproducible setup.
- When dependencies change:
  1. Update `pyproject.toml`.
  2. Reinstall in `.venv`.
  3. Run `make lock`.

## 3) Code quality gate
- Lint must pass: `make lint`.
- Tests must pass: `make test`.
- Combined gate before commit/merge: `make check`.

## 4) Data/feature pipeline commands
- Download raw OHLCV: `make run-download`
- Validate raw data: `make run-validate`
- Build phase-2 features: `make run-features`
- Build phase-3 labels: `make run-labels`
- Build phase-4 dataset split: `make run-dataset`
- Run post-split checks: `make run-dataset-checks`
- Train phase-5 baseline model: `make run-train-model1`
- Build phase-6 scanner report: `make run-scanner-report`
- Build phase-6 hotzones: `make run-hotzones`
- Run phase-6 leakage checks: `make run-leakage-checks`
- Build phase-7 scanner tuning summary: `make run-scanner-tuning-summary`
- Export phase-8 hotzones UI rectangles: `make run-export-hotzones-ui`
- Export phase-8 zoneRisk points: `make run-export-zonerisk-points`
- Run all phase-8 exports: `make run-phase8-ui`

## 5) No-lookahead rule
- Any feature at time `t` must not use data from `t+1` onward.
- Swing/pivot features must use confirmation delay consistently.

## 6) Artifacts and generated files
- Keep venv and caches out of git.
- Feature outputs in `artifacts/processed/` are generated artifacts.
- Label outputs in `artifacts/processed/` and `artifacts/reports/` are generated artifacts.
- Re-generate artifacts from scripts, do not hand-edit them.
- Phase 7 tuning summary artifact: `artifacts/reports/scanner_tuning_summary.json`.
- Phase 8 UI artifacts:
- `artifacts/reports/hotzones_ui.json`
- `artifacts/reports/zoneRisk_points.json`

## 7) Scanner freeze (v1)
- `hot_threshold=0.80`
- `min_zone_bars=2`
- `max_gap_bars=1`
- `report_thresholds=[0.60, 0.70, 0.75, 0.80, 0.85, 0.90]`

## 8) FE export contract (Phase 8)
- Hotzone rectangle contract:
- `from_time/to_time` for x-axis bounds.
- `bottom_price/top_price` for y-axis bounds.
- Keep time as ISO UTC string.
- zoneRisk overlay contract:
- points list of `{time, zoneRisk}` with `zoneRisk` in `[0,1]`.
