---
name: trading-feature-pipeline
description: Build and sanity-check phase-2 feature engineering outputs for this repository. Use when the user asks to generate, inspect, or debug features_core/features_structure parquet files or no-lookahead swing logic.
---

# Trading Feature Pipeline

Run feature engineering with repository defaults:

1. Activate virtual environment: `source .venv/bin/activate`.
2. Build features: `make run-features`.
3. Verify outputs:
   - `artifacts/processed/features_core.parquet`
   - `artifacts/processed/features_structure.parquet`

Run quick sanity checks after generation:

1. Confirm required core columns:
   `time, atr14, atr_pct, range1, ret1, ret3, ret6, abs_ret1, vol_sma50, vol_ratio`
2. Confirm required structure columns:
   `time, last_swing_high, last_swing_low, dist_high_atr, dist_low_atr, near_structure`
3. Confirm non-empty row counts and no critical null bursts in key columns.

Enforce no-lookahead behavior:

1. Treat pivot confirmation as delayed by `swing_size`.
2. Use only confirmed swings (`pivot_index + swing_size`) as available structure state.
3. Keep features aligned to bar `t` only.

When reporting results, include:

1. Output row counts and column lists.
2. Warmup and swing settings read from `configs/config.yaml`.
3. Any anomalies (NaN spikes, empty output, missing columns) with exact file path.

For concrete check snippets, read `references/checks.md`.
