# Commands

## Standard workflow

```bash
source .venv/bin/activate
make run-download
make run-validate
```

## Direct scripts

```bash
source .venv/bin/activate
python src/data/download_data.py
python src/data/validate_data.py
```

## Check outputs

```bash
ls -lh artifacts/raw/BTCUSDT_15m.csv artifacts/reports/data_quality.json
python - <<'PY'
import json
with open("artifacts/reports/data_quality.json", "r") as f:
    print(json.load(f))
PY
```
