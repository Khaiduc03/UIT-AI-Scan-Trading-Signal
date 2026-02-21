import json
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd
import yaml

THRESHOLD_TOL = 1e-9

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def _read_json(path: Path) -> dict:
    with open(path, "r") as file:
        return json.load(file)


def _find_threshold_row(report: dict, hot_threshold: float) -> dict:
    rows = report.get("thresholds", [])
    for row in rows:
        if abs(float(row.get("threshold")) - hot_threshold) <= THRESHOLD_TOL:
            return row
    raise ValueError(
        f"hot_threshold={hot_threshold:.6f} not found in scanner threshold report"
    )


def build_summary(
    scanner_cfg: dict,
    threshold_report: dict,
    hotzones_report: dict,
    test_df: pd.DataFrame,
    threshold_report_path: Path,
    hotzones_report_path: Path,
    test_path: Path,
    output_path: Path,
) -> dict:
    if test_df.empty:
        raise ValueError("test split is empty (test_rows == 0)")
    if "time" not in test_df.columns:
        raise ValueError("test split must contain `time` column")

    test_time = pd.to_datetime(test_df["time"], utc=True)
    start_time = test_time.min()
    end_time = test_time.max()
    test_rows = int(len(test_df))

    months_in_test = float((end_time - start_time).total_seconds() / 86400.0 / 30.0)
    if months_in_test <= 0:
        raise ValueError(f"invalid test time range: months_in_test={months_in_test}")

    hot_threshold = float(scanner_cfg.get("hot_threshold", 0.80))
    min_zone_bars = int(scanner_cfg.get("min_zone_bars", 2))
    max_gap_bars = int(scanner_cfg.get("max_gap_bars", 1))

    total_zones = int(hotzones_report.get("total_zones", 0))
    total_hot_bars = int(hotzones_report.get("total_hot_bars", 0))
    coverage_hot_bars = float(total_hot_bars / test_rows)
    zones_per_month = float(total_zones / months_in_test)

    quality = _find_threshold_row(threshold_report, hot_threshold)

    return {
        "params": {
            "hot_threshold": hot_threshold,
            "min_zone_bars": min_zone_bars,
            "max_gap_bars": max_gap_bars,
        },
        "test_range": {
            "start_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "test_rows": test_rows,
            "months_in_test": months_in_test,
        },
        "density": {
            "total_zones": total_zones,
            "zones_per_month": zones_per_month,
            "total_hot_bars": total_hot_bars,
            "coverage_hot_bars": coverage_hot_bars,
        },
        "quality_at_hot_threshold": {
            "threshold": hot_threshold,
            "flagged_bars": int(quality.get("flagged_bars", 0)),
            "coverage": float(quality.get("coverage", 0.0)),
            "hit_rate": quality.get("hit_rate"),
            "lift_vs_base": quality.get("lift_vs_base"),
            "hit_within_k_rate": quality.get("hit_within_k_rate"),
            "precision": quality.get("precision"),
            "recall": quality.get("recall"),
            "f1": quality.get("f1"),
        },
        "artifacts": {
            "threshold_report": str(threshold_report_path),
            "hotzones_report": str(hotzones_report_path),
            "test_split": str(test_path),
            "output_summary": str(output_path),
        },
    }


def run(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    scanner_cfg = cfg.get("scanner", {})
    split_cfg = cfg.get("split", {})
    processed_dir = Path(split_cfg.get("output_dir", "artifacts/processed"))

    threshold_report_path = Path(
        scanner_cfg.get("out_threshold_report", "artifacts/reports/scanner_threshold_report.json")
    )
    hotzones_report_path = Path(
        scanner_cfg.get("out_hotzones", "artifacts/reports/hotzones_test.json")
    )
    test_path = Path(split_cfg.get("out_test", processed_dir / "test.parquet"))
    output_path = Path(
        scanner_cfg.get("out_tuning_summary", "artifacts/reports/scanner_tuning_summary.json")
    )

    for p in [threshold_report_path, hotzones_report_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"missing required input file: {p}")

    threshold_report = _read_json(threshold_report_path)
    hotzones_report = _read_json(hotzones_report_path)
    test_df = pd.read_parquet(test_path)
    summary = build_summary(
        scanner_cfg=scanner_cfg,
        threshold_report=threshold_report,
        hotzones_report=hotzones_report,
        test_df=test_df,
        threshold_report_path=threshold_report_path,
        hotzones_report_path=hotzones_report_path,
        test_path=test_path,
        output_path=output_path,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(summary, file, indent=2)
    return output_path


def main() -> None:
    start_time = perf_counter()
    output_path = run()
    summary = _read_json(output_path)
    logger.info(
        "Scanner tuning summary saved: %s | zones=%s | zones_per_month=%.4f | "
        "coverage_hot_bars=%.4f | elapsed=%.2fs",
        output_path,
        summary["density"]["total_zones"],
        summary["density"]["zones_per_month"],
        summary["density"]["coverage_hot_bars"],
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()
