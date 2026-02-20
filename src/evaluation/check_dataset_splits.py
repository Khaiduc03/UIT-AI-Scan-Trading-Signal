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


def _read_paths(cfg: dict) -> dict[str, Path]:
    split_cfg = cfg.get("split", {})
    processed_dir = Path(split_cfg.get("output_dir", "artifacts/processed"))
    return {
        "dataset": Path(split_cfg.get("out_dataset", processed_dir / "dataset_model1.parquet")),
        "train": Path(split_cfg.get("out_train", processed_dir / "train.parquet")),
        "val": Path(split_cfg.get("out_val", processed_dir / "val.parquet")),
        "test": Path(split_cfg.get("out_test", processed_dir / "test.parquet")),
        "splits_meta": Path(split_cfg.get("out_splits", processed_dir / "splits.json")),
        "report": Path("artifacts/reports/dataset_checks.json"),
    }


def _must_exist(paths: dict[str, Path]) -> None:
    for key in ["dataset", "train", "val", "test", "splits_meta"]:
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing required file for checks: {paths[key]}")


def _class_balance(df: pd.DataFrame, target_col: str) -> dict:
    total = int(len(df))
    pos = int(df[target_col].sum()) if total else 0
    neg = total - pos
    return {
        "rows": total,
        "positive_count": pos,
        "negative_count": neg,
        "positive_rate": float(pos / total) if total else 0.0,
    }


def _drift_report(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> dict:
    numeric_cols = [
        c
        for c in train_df.select_dtypes(include=[np.number]).columns
        if c != target_col and c in test_df.columns
    ]
    rows = []
    for col in numeric_cols:
        mean_tr = float(train_df[col].mean())
        mean_te = float(test_df[col].mean())
        std_tr = float(train_df[col].std())
        std_te = float(test_df[col].std())
        mean_shift_sigma = float(abs(mean_te - mean_tr) / (std_tr + 1e-12))
        std_ratio = float(std_te / (std_tr + 1e-12))
        rows.append(
            {
                "feature": col,
                "mean_train": mean_tr,
                "mean_test": mean_te,
                "std_train": std_tr,
                "std_test": std_te,
                "mean_shift_sigma": mean_shift_sigma,
                "std_ratio_test_train": std_ratio,
            }
        )

    rows.sort(key=lambda r: r["mean_shift_sigma"], reverse=True)
    return {
        "feature_count_checked": len(rows),
        "top_mean_shift_features": rows[:10],
    }


def run_checks() -> dict:
    cfg = load_config()
    target_col = cfg.get("model1", {}).get("target_col", "StrongMove")
    paths = _read_paths(cfg)
    _must_exist(paths)

    dataset = pd.read_parquet(paths["dataset"])
    train_df = pd.read_parquet(paths["train"])
    val_df = pd.read_parquet(paths["val"])
    test_df = pd.read_parquet(paths["test"])
    with open(paths["splits_meta"], "r") as file:
        splits_meta = json.load(file)

    # Basic structural checks
    same_schema = (
        list(dataset.columns)
        == list(train_df.columns)
        == list(val_df.columns)
        == list(test_df.columns)
    )
    no_overlap = (
        not set(train_df["time"]).intersection(val_df["time"])
        and not set(train_df["time"]).intersection(test_df["time"])
        and not set(val_df["time"]).intersection(test_df["time"])
    )
    strict_order = train_df["time"].max() < val_df["time"].min() < test_df["time"].min()
    full_coverage = len(dataset) == len(train_df) + len(val_df) + len(test_df)

    metadata_counts_match = (
        int(splits_meta["splits"]["train"]["rows"]) == len(train_df)
        and int(splits_meta["splits"]["val"]["rows"]) == len(val_df)
        and int(splits_meta["splits"]["test"]["rows"]) == len(test_df)
        and int(splits_meta["total_rows"]) == len(dataset)
    )

    report = {
        "status": "pass",
        "checks": {
            "same_schema_all_splits": same_schema,
            "target_exists": target_col in dataset.columns,
            "time_sorted_dataset": bool(dataset["time"].is_monotonic_increasing),
            "no_overlap": no_overlap,
            "strict_time_order": strict_order,
            "full_coverage": full_coverage,
            "metadata_counts_match_files": metadata_counts_match,
        },
        "class_balance": {
            "dataset": _class_balance(dataset, target_col),
            "train": _class_balance(train_df, target_col),
            "val": _class_balance(val_df, target_col),
            "test": _class_balance(test_df, target_col),
        },
        "drift_train_vs_test": _drift_report(train_df, test_df, target_col),
        "paths": {k: str(v) for k, v in paths.items() if k != "report"},
    }

    if not all(report["checks"].values()):
        report["status"] = "fail"

    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["report"], "w") as file:
        json.dump(report, file, indent=2)

    return report


def main():
    start_time = perf_counter()
    report = run_checks()
    logger.info("Dataset checks status: %s", report["status"])
    logger.info("Check report saved at artifacts/reports/dataset_checks.json")
    logger.info(
        "Class balance train/val/test: %.4f / %.4f / %.4f",
        report["class_balance"]["train"]["positive_rate"],
        report["class_balance"]["val"]["positive_rate"],
        report["class_balance"]["test"]["positive_rate"],
    )
    top_shift = report["drift_train_vs_test"]["top_mean_shift_features"]
    if top_shift:
        logger.info(
            "Top drift feature: %s (mean_shift_sigma=%.4f)",
            top_shift[0]["feature"],
            top_shift[0]["mean_shift_sigma"],
        )
    logger.info("Dataset checks elapsed=%.2fs", perf_counter() - start_time)


if __name__ == "__main__":
    main()
