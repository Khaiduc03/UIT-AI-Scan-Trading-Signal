import json
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.export_zoneRisk_points import run


def test_export_zonerisk_points(tmp_path: Path):
    zone_risk_path = tmp_path / "reports" / "zoneRisk_test.parquet"
    out_points_path = tmp_path / "reports" / "zoneRisk_points.json"
    cfg_path = tmp_path / "configs" / "config.yaml"

    df = pd.DataFrame(
        {
            "time": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:15:00Z",
                "2026-01-01T00:30:00Z",
            ],
            "StrongMove": [0, 1, 0],
            "zoneRisk": [0.0, 0.5, 1.0],
        }
    )
    zone_risk_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(zone_risk_path, index=False)

    cfg = {
        "data": {"timeframe": "15m"},
        "scanner": {
            "out_zoneRisk_test": str(zone_risk_path),
            "out_zoneRisk_points": str(out_points_path),
        },
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as file:
        yaml.safe_dump(cfg, file)

    out = run(str(cfg_path))
    assert out == out_points_path
    assert out_points_path.exists()

    payload = json.load(open(out_points_path))
    assert payload["timeframe"] == "15m"
    assert payload["rows"] == len(df)
    assert len(payload["points"]) == len(df)
    for p in payload["points"]:
        assert 0.0 <= p["zoneRisk"] <= 1.0

