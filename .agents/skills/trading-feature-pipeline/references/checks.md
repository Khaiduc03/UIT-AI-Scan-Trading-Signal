# Checks

## Build features

```bash
source .venv/bin/activate
make run-features
```

## Validate schema and shape

```bash
source .venv/bin/activate
python - <<'PY'
import pandas as pd

core = pd.read_parquet("artifacts/processed/features_core.parquet")
structure = pd.read_parquet("artifacts/processed/features_structure.parquet")

print("core rows:", len(core))
print("core cols:", core.columns.tolist())
print("structure rows:", len(structure))
print("structure cols:", structure.columns.tolist())
PY
```

## Basic null scan

```bash
source .venv/bin/activate
python - <<'PY'
import pandas as pd

core = pd.read_parquet("artifacts/processed/features_core.parquet")
print(core[["atr14", "ret1", "vol_ratio"]].isna().sum())
PY
```
