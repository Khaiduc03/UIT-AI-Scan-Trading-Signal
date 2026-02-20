---
name: trading-data-pipeline
description: Run and verify BTCUSDT OHLCV ingestion for this repository. Use when the user asks to download, refresh, validate, or troubleshoot raw market data in artifacts/raw and artifacts/reports.
---

# Trading Data Pipeline

Run the repository's data pipeline in this order:

1. Activate virtual environment: `source .venv/bin/activate`.
2. Download data: `make run-download`.
3. Validate data: `make run-validate`.
4. Read report: `artifacts/reports/data_quality.json`.

If `make` is unavailable, run scripts directly:

- `python src/data/download_data.py`
- `python src/data/validate_data.py`

When reporting results, include:

1. Input symbol, timeframe, and date range from `configs/config.yaml`.
2. Row count and first/last timestamp from `artifacts/raw/BTCUSDT_15m.csv`.
3. Validation status and errors from `artifacts/reports/data_quality.json`.

If download fails:

1. Confirm internet access and exchange name in config.
2. Confirm symbol format required by CCXT (`BTC/USDT`).
3. Retry once, then report the exact exception and failing step.

For quick command references, read `references/commands.md`.
