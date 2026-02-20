import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def build_hotzones(
    zone_df: pd.DataFrame,
    hot_threshold: float,
    min_zone_bars: int,
    max_gap_bars: int,
) -> list[dict]:
    if zone_df.empty:
        return []

    df = zone_df.sort_values("time").reset_index(drop=True)
    hot_idx = np.where(df["zoneRisk"].to_numpy(dtype=float) >= hot_threshold)[0]
    if len(hot_idx) == 0:
        return []

    clusters: list[tuple[int, int, list[int]]] = []
    cluster_hot = [int(hot_idx[0])]
    for idx in hot_idx[1:]:
        prev = cluster_hot[-1]
        gap = int(idx - prev - 1)
        if gap <= max_gap_bars:
            cluster_hot.append(int(idx))
        else:
            clusters.append((cluster_hot[0], cluster_hot[-1], cluster_hot.copy()))
            cluster_hot = [int(idx)]
    clusters.append((cluster_hot[0], cluster_hot[-1], cluster_hot.copy()))

    zones = []
    zone_id = 1
    for start_i, end_i, hot_members in clusters:
        count_hot = len(hot_members)
        if count_hot < min_zone_bars:
            continue
        slice_df = df.iloc[start_i : end_i + 1]
        zones.append(
            {
                "zone_id": zone_id,
                "from_time": str(slice_df["time"].iloc[0]),
                "to_time": str(slice_df["time"].iloc[-1]),
                "from_index": int(start_i),
                "to_index": int(end_i),
                "max_risk": float(slice_df["zoneRisk"].max()),
                "avg_risk": float(slice_df["zoneRisk"].mean()),
                "count_hot_bars": int(count_hot),
                "count_bars_total": int(end_i - start_i + 1),
            }
        )
        zone_id += 1

    return zones


def main():
    start_time = perf_counter()

    cfg = load_config()
    scanner_cfg = cfg.get("scanner", {})

    in_zone_path = Path(
        scanner_cfg.get("out_zoneRisk_test", "artifacts/reports/zoneRisk_test.parquet")
    )
    out_hotzones = Path(scanner_cfg.get("out_hotzones", "artifacts/reports/hotzones_test.json"))
    hot_threshold = float(scanner_cfg.get("hot_threshold", 0.75))
    min_zone_bars = int(scanner_cfg.get("min_zone_bars", 2))
    max_gap_bars = int(scanner_cfg.get("max_gap_bars", 1))

    if not in_zone_path.exists():
        raise FileNotFoundError(f"Missing zoneRisk series: {in_zone_path}")

    logger.info("Loading zoneRisk series: %s", in_zone_path)
    zone_df = pd.read_parquet(in_zone_path)
    zones = build_hotzones(
        zone_df=zone_df,
        hot_threshold=hot_threshold,
        min_zone_bars=min_zone_bars,
        max_gap_bars=max_gap_bars,
    )

    total_hot_bars = int((zone_df["zoneRisk"].to_numpy(dtype=float) >= hot_threshold).sum())
    output = {
        "hot_threshold": hot_threshold,
        "min_zone_bars": min_zone_bars,
        "max_gap_bars": max_gap_bars,
        "total_zones": len(zones),
        "total_hot_bars": total_hot_bars,
        "zones": zones,
    }

    out_hotzones.parent.mkdir(parents=True, exist_ok=True)
    with open(out_hotzones, "w") as file:
        json.dump(output, file, indent=2)

    logger.info("Hotzones saved: %s (zones=%s)", out_hotzones, len(zones))
    logger.info(
        "Hotzone summary | hot_threshold=%.2f | min_zone_bars=%s | "
        "max_gap_bars=%s | total_hot_bars=%s | elapsed=%.2fs",
        hot_threshold,
        min_zone_bars,
        max_gap_bars,
        total_hot_bars,
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()
