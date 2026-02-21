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


def _read_json(path: Path) -> dict:
    with open(path, "r") as file:
        return json.load(file)


def _validate_unique_time(df: pd.DataFrame, name: str) -> None:
    if "time" not in df.columns:
        raise ValueError(f"{name} must contain `time` column")
    if not df["time"].is_unique:
        raise ValueError(f"{name} has duplicate `time` values")


def _zone_slice_from_index(
    test_ohlc: pd.DataFrame,
    zone: dict,
) -> tuple[pd.DataFrame, int, int] | None:
    from_index = zone.get("from_index")
    to_index = zone.get("to_index")
    if from_index is None or to_index is None:
        return None
    try:
        start = int(from_index)
        end = int(to_index)
    except (TypeError, ValueError):
        return None
    if start < 0 or end < 0 or start > end or end >= len(test_ohlc):
        return None
    return test_ohlc.iloc[start : end + 1], start, end


def _zone_slice_from_time(test_ohlc: pd.DataFrame, zone: dict) -> tuple[pd.DataFrame, int, int]:
    from_time = zone.get("from_time")
    to_time = zone.get("to_time")
    if not from_time or not to_time:
        raise ValueError("zone must include from_time and to_time for fallback slicing")
    mask = (test_ohlc["time"] >= from_time) & (test_ohlc["time"] <= to_time)
    sliced = test_ohlc.loc[mask]
    if sliced.empty:
        raise ValueError(f"zone slice is empty for time range [{from_time}, {to_time}]")
    return sliced, int(sliced.index.min()), int(sliced.index.max())


def build_hotzones_ui(
    hotzones_report: dict,
    zone_risk_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    timeframe: str,
) -> dict:
    _validate_unique_time(zone_risk_df, "zoneRisk_test")
    _validate_unique_time(raw_df, "raw")

    required_raw_cols = {"time", "high", "low"}
    if not required_raw_cols.issubset(raw_df.columns):
        raise ValueError(f"raw is missing required columns: {sorted(required_raw_cols)}")

    test_ohlc = zone_risk_df.merge(
        raw_df[["time", "high", "low"]],
        on="time",
        how="inner",
        validate="one_to_one",
    )
    test_ohlc = test_ohlc.sort_values("time").reset_index(drop=True)
    if test_ohlc.empty:
        raise ValueError("merged test_ohlc is empty")

    zones_ui: list[dict] = []
    for zone in hotzones_report.get("zones", []):
        sliced_info = _zone_slice_from_index(test_ohlc, zone)
        if sliced_info is None:
            sliced, start_idx, end_idx = _zone_slice_from_time(test_ohlc, zone)
        else:
            sliced, start_idx, end_idx = sliced_info

        top_price = float(sliced["high"].max())
        bottom_price = float(sliced["low"].min())
        if top_price < bottom_price:
            raise ValueError(
                f"invalid zone price range for zone_id={zone.get('zone_id')}: "
                f"top_price({top_price}) < bottom_price({bottom_price})"
            )

        zones_ui.append(
            {
                "zone_id": int(zone["zone_id"]),
                "from_time": str(zone["from_time"]),
                "to_time": str(zone["to_time"]),
                "from_index": int(start_idx),
                "to_index": int(end_idx),
                "top_price": top_price,
                "bottom_price": bottom_price,
                "mid_price": float((top_price + bottom_price) / 2.0),
                "max_risk": float(zone["max_risk"]),
                "avg_risk": float(zone["avg_risk"]),
                "count_hot_bars": int(zone["count_hot_bars"]),
                "count_bars_total": int(zone["count_bars_total"]),
            }
        )

    scanner_params = {
        "hot_threshold": float(hotzones_report.get("hot_threshold")),
        "min_zone_bars": int(hotzones_report.get("min_zone_bars")),
        "max_gap_bars": int(hotzones_report.get("max_gap_bars")),
    }
    return {
        "timeframe": timeframe,
        "params": scanner_params,
        "test_range": {
            "start_time": str(test_ohlc["time"].iloc[0]),
            "end_time": str(test_ohlc["time"].iloc[-1]),
            "rows": int(len(test_ohlc)),
        },
        "total_zones": int(len(zones_ui)),
        "zones": zones_ui,
    }


def run(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})
    scanner_cfg = cfg.get("scanner", {})

    processed_dir = Path(split_cfg.get("output_dir", "artifacts/processed"))
    raw_path = Path(data_cfg.get("output_csv", "artifacts/raw/BTCUSDT_15m.csv"))
    zone_risk_path = Path(
        scanner_cfg.get("out_zoneRisk_test", "artifacts/reports/zoneRisk_test.parquet")
    )
    hotzones_path = Path(scanner_cfg.get("out_hotzones", "artifacts/reports/hotzones_test.json"))
    out_ui_path = Path(scanner_cfg.get("out_hotzones_ui", "artifacts/reports/hotzones_ui.json"))
    if not raw_path.exists():
        raise FileNotFoundError(f"missing raw candles file: {raw_path}")
    if not zone_risk_path.exists():
        raise FileNotFoundError(f"missing zoneRisk file: {zone_risk_path}")
    if not hotzones_path.exists():
        raise FileNotFoundError(f"missing hotzones file: {hotzones_path}")
    if not processed_dir.exists():
        logger.warning(
            "processed_dir not found (%s), continuing with available inputs",
            processed_dir,
        )

    raw_df = pd.read_csv(raw_path)
    zone_risk_df = pd.read_parquet(zone_risk_path)
    hotzones_report = _read_json(hotzones_path)
    output = build_hotzones_ui(
        hotzones_report=hotzones_report,
        zone_risk_df=zone_risk_df,
        raw_df=raw_df,
        timeframe=data_cfg.get("timeframe", "15m"),
    )

    out_ui_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ui_path, "w") as file:
        json.dump(output, file, indent=2)
    return out_ui_path


def main() -> None:
    start_time = perf_counter()
    out_path = run()
    output = _read_json(out_path)
    logger.info(
        "Hotzones UI export saved: %s | zones=%s | rows=%s | elapsed=%.2fs",
        out_path,
        output["total_zones"],
        output["test_range"]["rows"],
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()
