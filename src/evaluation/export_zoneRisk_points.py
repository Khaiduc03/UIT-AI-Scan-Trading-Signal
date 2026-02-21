import json
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def build_points(zone_risk_df: pd.DataFrame, timeframe: str) -> dict:
    required = {"time", "zoneRisk"}
    if not required.issubset(zone_risk_df.columns):
        raise ValueError(f"zoneRisk dataframe missing required columns: {sorted(required)}")

    points = []
    for row in zone_risk_df[["time", "zoneRisk"]].itertuples(index=False):
        risk = float(row.zoneRisk)
        if risk < 0.0 or risk > 1.0:
            raise ValueError(f"zoneRisk outside [0,1]: {risk}")
        points.append({"time": str(row.time), "zoneRisk": risk})

    return {
        "timeframe": timeframe,
        "rows": int(len(points)),
        "points": points,
    }


def run(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    scanner_cfg = cfg.get("scanner", {})
    in_path = Path(scanner_cfg.get("out_zoneRisk_test", "artifacts/reports/zoneRisk_test.parquet"))
    out_path = Path(
        scanner_cfg.get("out_zoneRisk_points", "artifacts/reports/zoneRisk_points.json")
    )

    if not in_path.exists():
        raise FileNotFoundError(f"missing zoneRisk input file: {in_path}")
    zone_risk_df = pd.read_parquet(in_path).sort_values("time").reset_index(drop=True)
    output = build_points(zone_risk_df=zone_risk_df, timeframe=data_cfg.get("timeframe", "15m"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as file:
        json.dump(output, file, indent=2)
    return out_path


def main() -> None:
    start_time = perf_counter()
    out_path = run()
    with open(out_path, "r") as file:
        output = json.load(file)
    logger.info(
        "zoneRisk points export saved: %s | rows=%s | elapsed=%.2fs",
        out_path,
        output["rows"],
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()
