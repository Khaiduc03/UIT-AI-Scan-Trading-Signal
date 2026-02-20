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

## 5) No-lookahead rule
- Any feature at time `t` must not use data from `t+1` onward.
- Swing/pivot features must use confirmation delay consistently.

## 6) Artifacts and generated files
- Keep venv and caches out of git.
- Feature outputs in `artifacts/processed/` are generated artifacts.
- Label outputs in `artifacts/processed/` and `artifacts/reports/` are generated artifacts.
- Re-generate artifacts from scripts, do not hand-edit them.
