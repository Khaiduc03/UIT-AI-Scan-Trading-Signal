import json
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.export_hotzones_ui import run


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def test_export_hotzones_ui(tmp_path: Path):
    raw_path = tmp_path / "raw" / "BTCUSDT_15m.csv"
    zone_risk_path = tmp_path / "reports" / "zoneRisk_test.parquet"
    hotzones_path = tmp_path / "reports" / "hotzones_test.json"
    out_ui_path = tmp_path / "reports" / "hotzones_ui.json"
    cfg_path = tmp_path / "configs" / "config.yaml"

    times = pd.date_range("2026-01-01", periods=6, freq="15min", tz="UTC")
    time_vals = times.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()

    raw_df = pd.DataFrame(
        {
            "time": time_vals,
            "open": [100, 101, 102, 103, 104, 105],
            "high": [110, 112, 111, 115, 116, 117],
            "low": [99, 100, 98, 102, 101, 100],
            "close": [101, 102, 103, 104, 105, 106],
            "volume": [10, 20, 30, 40, 50, 60],
        }
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)

    zone_risk_df = pd.DataFrame(
        {
            "time": time_vals,
            "StrongMove": [0, 1, 0, 1, 0, 1],
            "zoneRisk": [0.2, 0.81, 0.85, 0.4, 0.9, 0.91],
        }
    )
    zone_risk_path.parent.mkdir(parents=True, exist_ok=True)
    zone_risk_df.to_parquet(zone_risk_path, index=False)

    hotzones_report = {
        "hot_threshold": 0.8,
        "min_zone_bars": 2,
        "max_gap_bars": 1,
        "total_zones": 2,
        "total_hot_bars": 4,
        "zones": [
            {
                "zone_id": 1,
                "from_time": time_vals[1],
                "to_time": time_vals[2],
                "from_index": 1,
                "to_index": 2,
                "max_risk": 0.85,
                "avg_risk": 0.83,
                "count_hot_bars": 2,
                "count_bars_total": 2,
            },
            {
                "zone_id": 2,
                "from_time": time_vals[4],
                "to_time": time_vals[5],
                "from_index": 4,
                "to_index": 5,
                "max_risk": 0.91,
                "avg_risk": 0.905,
                "count_hot_bars": 2,
                "count_bars_total": 2,
            },
        ],
    }
    _write_json(hotzones_path, hotzones_report)

    cfg = {
        "data": {
            "timeframe": "15m",
            "output_csv": str(raw_path),
        },
        "split": {
            "output_dir": str(tmp_path / "processed"),
        },
        "scanner": {
            "out_zoneRisk_test": str(zone_risk_path),
            "out_hotzones": str(hotzones_path),
            "out_hotzones_ui": str(out_ui_path),
        },
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as file:
        yaml.safe_dump(cfg, file)

    out = run(str(cfg_path))
    assert out == out_ui_path
    assert out_ui_path.exists()

    payload = json.load(open(out_ui_path))
    zones = payload["zones"]
    assert payload["total_zones"] == len(zones)
    assert len(zones) == hotzones_report["total_zones"]

    rows = payload["test_range"]["rows"]
    for zone in zones:
        assert zone["top_price"] >= zone["bottom_price"]
        assert 0 <= zone["from_index"] <= zone["to_index"] <= (rows - 1)
        assert zone["from_time"] <= zone["to_time"]

