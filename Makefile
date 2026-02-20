PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup dev lock check lint format test run-download run-validate run-features run-labels run-dataset run-dataset-checks run-train-model1 run-scanner-report run-hotzones run-leakage-checks run-all-phases summary-features summary-labels summary-dataset summary-train summary-scanner summary-hotzones summary-leakage clean

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e .

dev: setup
	$(PIP) install -e ".[dev]"

lock:
	$(PIP) freeze | rg -v "^-e " > requirements.txt

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

test:
	$(PYTHON) -m pytest -q tests

check: lint test

run-download:
	@echo "[run-download] Download BTC OHLCV raw data from exchange into artifacts/raw."
	@RAW_CSV=`$(PYTHON) -c "import yaml;cfg=yaml.safe_load(open('configs/config.yaml'));print(cfg.get('data',{}).get('output_csv','artifacts/raw/BTCUSDT_15m.csv'))"`; \
	if [ -f "$$RAW_CSV" ]; then \
		echo "[run-download] Existing raw file detected: $$RAW_CSV"; \
		if [ "$$REDOWNLOAD" = "1" ]; then \
			echo "[run-download] REDOWNLOAD=1 -> re-downloading and overwriting."; \
			$(PYTHON) src/data/download_data.py; \
		elif [ "$$REDOWNLOAD" = "0" ]; then \
			echo "[run-download] REDOWNLOAD=0 -> skip download."; \
		elif [ -t 0 ]; then \
			printf "[run-download] Re-download and overwrite? [y/N]: "; \
			read ans; \
			case "$$ans" in \
				y|Y|yes|YES) $(PYTHON) src/data/download_data.py ;; \
				*) echo "[run-download] Skip download, keep existing file." ;; \
			esac; \
		else \
			echo "[run-download] Non-interactive shell -> skip download. Use REDOWNLOAD=1 make run-download to force."; \
		fi; \
	else \
		echo "[run-download] No raw file found -> downloading now."; \
		$(PYTHON) src/data/download_data.py; \
	fi

run-validate:
	@echo "[run-validate] Validate raw OHLCV quality (columns/sort/NaN/duplicates)."
	$(PYTHON) src/data/validate_data.py

run-features:
	@echo "[run-features] Build Phase 2 features_core and features_structure."
	$(PYTHON) src/features/build_features.py

run-labels:
	@echo "[run-labels] Build Phase 3 StrongMove labels and distribution report."
	$(PYTHON) src/labels/build_labels.py

run-dataset:
	@echo "[run-dataset] Build Phase 4 dataset_model1 and time-based train/val/test splits."
	$(PYTHON) src/models/build_dataset.py

run-dataset-checks:
	@echo "[run-dataset-checks] Check split integrity, class balance, and train-vs-test drift."
	$(PYTHON) src/evaluation/check_dataset_splits.py

run-train-model1:
	@echo "[run-train-model1] Train Phase 5 baseline model and export model + metrics artifacts."
	$(PYTHON) src/models/train_model1.py

run-scanner-report:
	@echo "[run-scanner-report] Build Phase 6 zoneRisk series + threshold quality report."
	$(PYTHON) src/evaluation/run_scanner_report.py

run-hotzones:
	@echo "[run-hotzones] Extract Phase 6 hot zones with gap-aware grouping."
	$(PYTHON) src/evaluation/extract_hotzones.py

run-leakage-checks:
	@echo "[run-leakage-checks] Run Phase 6 leakage and cross-artifact sanity checks."
	$(PYTHON) src/evaluation/leakage_checks_phase6.py

run-all-phases:
	@echo "[run-all-phases] Running full pipeline Phase 1 -> Phase 6."
	@echo "==================== Phase 1: Data Ingestion ===================="
	$(MAKE) run-download
	$(MAKE) run-validate
	@echo "==================== Phase 2: Feature Engineering ===================="
	$(MAKE) run-features
	$(MAKE) summary-features
	@echo "==================== Phase 3: Labeling ===================="
	$(MAKE) run-labels
	$(MAKE) summary-labels
	@echo "==================== Phase 4: Dataset + Split ===================="
	$(MAKE) run-dataset
	$(MAKE) run-dataset-checks
	$(MAKE) summary-dataset
	@echo "==================== Phase 5: Model Training ===================="
	$(MAKE) run-train-model1
	$(MAKE) summary-train
	@echo "==================== Phase 6: Scanner + Hotzones + Leakage ===================="
	$(MAKE) run-scanner-report
	$(MAKE) summary-scanner
	$(MAKE) run-hotzones
	$(MAKE) summary-hotzones
	$(MAKE) run-leakage-checks
	$(MAKE) summary-leakage
	@echo "[run-all-phases] Completed."

summary-features:
	@printf '%s\n' \
		"import pandas as pd" \
		"core = pd.read_parquet('artifacts/processed/features_core.parquet')" \
		"struct = pd.read_parquet('artifacts/processed/features_structure.parquet')" \
		"print(f'[summary-features] core_rows={len(core)} structure_rows={len(struct)}')" \
		"print(f\"[summary-features] core_time={core['time'].iloc[0]} -> {core['time'].iloc[-1]}\")" | $(PYTHON)

summary-labels:
	@printf '%s\n' \
		"import pandas as pd" \
		"lb = pd.read_parquet('artifacts/processed/labels_strongmove.parquet')" \
		"rate = float(lb['StrongMove'].mean()) if len(lb) else 0.0" \
		"print(f'[summary-labels] rows={len(lb)} positive_rate={rate:.4f}')" \
		"print(f\"[summary-labels] time={lb['time'].iloc[0]} -> {lb['time'].iloc[-1]}\")" | $(PYTHON)

summary-dataset:
	@printf '%s\n' \
		"import json, pandas as pd" \
		"d = pd.read_parquet('artifacts/processed/dataset_model1.parquet')" \
		"m = json.load(open('artifacts/processed/splits.json'))" \
		"print(f'[summary-dataset] dataset_rows={len(d)} cols={len(d.columns)}')" \
		"print(f\"[summary-dataset] splits train/val/test = {m['splits']['train']['rows']}/{m['splits']['val']['rows']}/{m['splits']['test']['rows']}\")" \
		"print(f\"[summary-dataset] integrity={m['integrity_checks']}\")" | $(PYTHON)

summary-train:
	@printf '%s\n' \
		"import json" \
		"m = json.load(open('artifacts/reports/model1_metrics.json'))" \
		"print(f\"[summary-train] threshold={m['threshold_selection']['selected_threshold']:.4f}\")" \
		"print(f\"[summary-train] val pr_auc={m['val']['pr_auc']:.4f} | test pr_auc={m['test']['pr_auc']:.4f}\")" \
		"print(f\"[summary-train] test f1={m['test']['f1']:.4f} precision={m['test']['precision']:.4f} recall={m['test']['recall']:.4f}\")" | $(PYTHON)

summary-scanner:
	@printf '%s\n' \
		"import json" \
		"r = json.load(open('artifacts/reports/scanner_threshold_report.json'))" \
		"print('[summary-scanner] base_rate_test={:.4f}'.format(r['base_rate_test']))" \
		"for row in r['thresholds']:" \
		"    hr = 'null' if row['hit_rate'] is None else '{:.4f}'.format(row['hit_rate'])" \
		"    lift = 'null' if row['lift_vs_base'] is None else '{:.3f}'.format(row['lift_vs_base'])" \
		"    print('[summary-scanner] thr={:.2f} coverage={:.4f} hit_rate={} lift={} flagged={}'.format(row['threshold'], row['coverage'], hr, lift, row['flagged_bars']))" | $(PYTHON)

summary-hotzones:
	@printf '%s\n' \
		"import json" \
		"h = json.load(open('artifacts/reports/hotzones_test.json'))" \
		"print(f\"[summary-hotzones] zones={h['total_zones']} total_hot_bars={h['total_hot_bars']} threshold={h['hot_threshold']}\")" \
		"if h['zones']:" \
		"    z = h['zones'][0]" \
		"    print(f\"[summary-hotzones] first_zone id={z['zone_id']} from={z['from_time']} to={z['to_time']} count_hot={z['count_hot_bars']}\")" | $(PYTHON)

summary-leakage:
	@printf '%s\n' \
		"from pathlib import Path" \
		"p = Path('artifacts/reports/leakage_checks.md')" \
		"print(f'[summary-leakage] report={p} exists={p.exists()}')" \
		"text = p.read_text(encoding='utf-8') if p.exists() else ''" \
		"first = [line for line in text.splitlines() if line.startswith('- Overall status:')]" \
		"if first:" \
		"    print(f'[summary-leakage] {first[0]}')" | $(PYTHON)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
