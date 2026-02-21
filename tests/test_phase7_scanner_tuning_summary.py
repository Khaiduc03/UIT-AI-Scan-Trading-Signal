import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.evaluation.write_scanner_tuning_summary import run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def _write_config(
    path: Path,
    test_path: Path,
    threshold_path: Path,
    hotzones_path: Path,
    out_path: Path,
    hot_threshold: float = 0.80,
) -> None:
    payload = {
        "split": {
            "output_dir": str(test_path.parent),
            "out_test": str(test_path),
        },
        "scanner": {
            "hot_threshold": hot_threshold,
            "min_zone_bars": 2,
            "max_gap_bars": 1,
            "out_threshold_report": str(threshold_path),
            "out_hotzones": str(hotzones_path),
            "out_tuning_summary": str(out_path),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        yaml.safe_dump(payload, file)


def test_phase7_summary_happy_path_and_schema(tmp_path: Path):
    test_path = tmp_path / "processed" / "test.parquet"
    threshold_path = tmp_path / "reports" / "scanner_threshold_report.json"
    hotzones_path = tmp_path / "reports" / "hotzones_test.json"
    out_path = tmp_path / "reports" / "scanner_tuning_summary.json"
    cfg_path = tmp_path / "configs" / "config.yaml"

    test_df = pd.DataFrame(
        {
            "time": [
                "2025-10-10T01:30:00Z",
                "2025-11-10T01:30:00Z",
                "2025-12-10T01:30:00Z",
                "2026-01-10T01:30:00Z",
            ]
        }
    )
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(test_path, index=False)

    _write_json(
        threshold_path,
        {
            "thresholds": [
                {"threshold": 0.75, "flagged_bars": 10, "coverage": 0.10, "hit_rate": 0.8},
                {
                    "threshold": 0.80,
                    "flagged_bars": 8,
                    "coverage": 0.08,
                    "hit_rate": 0.9,
                    "lift_vs_base": 1.2,
                    "hit_within_k_rate": 0.95,
                    "precision": 0.9,
                    "recall": 0.4,
                    "f1": 0.55,
                },
                {"threshold": 0.85, "flagged_bars": 5, "coverage": 0.05, "hit_rate": 0.92},
            ]
        },
    )
    _write_json(
        hotzones_path,
        {
            "hot_threshold": 0.80,
            "min_zone_bars": 2,
            "max_gap_bars": 1,
            "total_zones": 6,
            "total_hot_bars": 3,
            "zones": [],
        },
    )
    _write_config(cfg_path, test_path, threshold_path, hotzones_path, out_path)

    created = run(str(cfg_path))
    assert created == out_path
    assert out_path.exists()

    summary = json.load(open(out_path))
    assert set(summary.keys()) >= {
        "params",
        "test_range",
        "density",
        "quality_at_hot_threshold",
        "artifacts",
    }
    assert set(summary["params"].keys()) == {"hot_threshold", "min_zone_bars", "max_gap_bars"}
    assert set(summary["test_range"].keys()) == {
        "start_time",
        "end_time",
        "test_rows",
        "months_in_test",
    }
    assert set(summary["density"].keys()) == {
        "total_zones",
        "zones_per_month",
        "total_hot_bars",
        "coverage_hot_bars",
    }
    assert set(summary["quality_at_hot_threshold"].keys()) == {
        "threshold",
        "flagged_bars",
        "coverage",
        "hit_rate",
        "lift_vs_base",
        "hit_within_k_rate",
        "precision",
        "recall",
        "f1",
    }

    assert summary["quality_at_hot_threshold"]["threshold"] == 0.8
    assert summary["quality_at_hot_threshold"]["flagged_bars"] == 8
    assert summary["density"]["zones_per_month"] > 0
    assert 0 <= summary["density"]["coverage_hot_bars"] <= 1


def test_phase7_missing_hot_threshold_row_raises(tmp_path: Path):
    test_path = tmp_path / "processed" / "test.parquet"
    threshold_path = tmp_path / "reports" / "scanner_threshold_report.json"
    hotzones_path = tmp_path / "reports" / "hotzones_test.json"
    out_path = tmp_path / "reports" / "scanner_tuning_summary.json"
    cfg_path = tmp_path / "configs" / "config.yaml"

    test_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": ["2025-10-10T01:30:00Z", "2025-10-11T01:30:00Z"]}).to_parquet(
        test_path,
        index=False,
    )
    _write_json(threshold_path, {"thresholds": [{"threshold": 0.75, "flagged_bars": 1}]})
    _write_json(hotzones_path, {"total_zones": 1, "total_hot_bars": 1, "zones": []})
    _write_config(cfg_path, test_path, threshold_path, hotzones_path, out_path, hot_threshold=0.80)

    with pytest.raises(ValueError, match="hot_threshold"):
        run(str(cfg_path))


def test_phase7_invalid_time_range_raises(tmp_path: Path):
    test_path = tmp_path / "processed" / "test.parquet"
    threshold_path = tmp_path / "reports" / "scanner_threshold_report.json"
    hotzones_path = tmp_path / "reports" / "hotzones_test.json"
    out_path = tmp_path / "reports" / "scanner_tuning_summary.json"
    cfg_path = tmp_path / "configs" / "config.yaml"

    test_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": ["2025-10-10T01:30:00Z"]}).to_parquet(test_path, index=False)
    _write_json(
        threshold_path,
        {
            "thresholds": [
                {
                    "threshold": 0.80,
                    "flagged_bars": 1,
                    "coverage": 1.0,
                    "hit_rate": 1.0,
                    "lift_vs_base": 1.0,
                    "hit_within_k_rate": 1.0,
                }
            ]
        },
    )
    _write_json(hotzones_path, {"total_zones": 1, "total_hot_bars": 1, "zones": []})
    _write_config(cfg_path, test_path, threshold_path, hotzones_path, out_path)

    with pytest.raises(ValueError, match="months_in_test"):
        run(str(cfg_path))
