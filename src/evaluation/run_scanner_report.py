import json
import logging
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def _timeframe_to_minutes(timeframe: str) -> int:
    if timeframe.endswith("m"):
        return int(timeframe[:-1])
    if timeframe.endswith("h"):
        return int(timeframe[:-1]) * 60
    raise ValueError(f"Unsupported timeframe format: {timeframe}")


def predict_zone_risk(test_df: pd.DataFrame, model_artifact: dict, target_col: str) -> pd.DataFrame:
    feature_columns = model_artifact["feature_columns"]
    if any(col not in test_df.columns for col in feature_columns):
        missing = [col for col in feature_columns if col not in test_df.columns]
        raise ValueError(f"Test split missing model feature columns: {missing}")
    if target_col not in test_df.columns:
        raise ValueError(f"Test split missing target column: {target_col}")

    probs = model_artifact["pipeline"].predict_proba(test_df[feature_columns])[:, 1]
    out = pd.DataFrame(
        {
            "time": test_df["time"],
            target_col: test_df[target_col].astype("int8"),
            "zoneRisk": probs.astype(float),
        }
    )
    return out.sort_values("time").reset_index(drop=True)


def _hit_within_k_rate(y_true: np.ndarray, flags: np.ndarray, horizon_k: int) -> float | None:
    idx = np.where(flags)[0]
    if len(idx) == 0:
        return None
    hits = []
    n = len(y_true)
    for t in idx:
        start = t + 1
        end = min(t + horizon_k, n - 1)
        if start > end:
            hits.append(0)
            continue
        hits.append(1 if np.any(y_true[start : end + 1] == 1) else 0)
    return float(np.mean(hits))


def threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
    horizon_k: int,
    base_rate: float,
) -> list[dict]:
    total = len(y_true)
    rows = []
    for thr in thresholds:
        flags = y_prob >= thr
        flagged = int(flags.sum())

        tp = int(np.sum((flags == 1) & (y_true == 1)))
        fp = int(np.sum((flags == 1) & (y_true == 0)))
        tn = int(np.sum((flags == 0) & (y_true == 0)))
        fn = int(np.sum((flags == 0) & (y_true == 1)))

        if flagged == 0:
            precision = None
            recall = None
            f1 = None
            hit_rate = None
            lift = None
            hit_within_k = None
            note = "no flagged bars at this threshold"
        else:
            preds = flags.astype(int)
            precision = float(precision_score(y_true, preds, zero_division=0))
            recall = float(recall_score(y_true, preds, zero_division=0))
            f1 = float(f1_score(y_true, preds, zero_division=0))
            hit_rate = float(tp / flagged)
            lift = float(hit_rate / base_rate) if base_rate > 0 else None
            hit_within_k = _hit_within_k_rate(y_true, flags, horizon_k)
            note = None

        row = {
            "threshold": float(thr),
            "flagged_bars": flagged,
            "coverage": float(flagged / total),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "hit_rate": hit_rate,
            "lift_vs_base": lift,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hit_within_k_rate": hit_within_k,
        }
        if note:
            row["note"] = note
        rows.append(row)
    return rows


def main():
    start_time = perf_counter()

    cfg = load_config()
    split_cfg = cfg.get("split", {})
    model_cfg = cfg.get("model1", {})
    scanner_cfg = cfg.get("scanner", {})
    label_cfg = cfg.get("label", {})
    data_cfg = cfg.get("data", {})

    processed_dir = Path(split_cfg.get("output_dir", "artifacts/processed"))
    test_path = Path(split_cfg.get("out_test", processed_dir / "test.parquet"))
    model_path = Path(model_cfg.get("output_path", "artifacts/models/model1.pkl"))
    out_zone_path = Path(
        scanner_cfg.get("out_zoneRisk_test", "artifacts/reports/zoneRisk_test.parquet")
    )
    out_report_path = Path(
        scanner_cfg.get("out_threshold_report", "artifacts/reports/scanner_threshold_report.json")
    )

    thresholds = [float(x) for x in scanner_cfg.get("report_thresholds", [0.6, 0.7, 0.75, 0.8])]
    horizon_k = int(label_cfg.get("horizon_k", 12))
    timeframe = data_cfg.get("timeframe", "15m")
    target_col = model_cfg.get("target_col", "StrongMove")

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    logger.info("Loading test split: %s", test_path)
    test_df = pd.read_parquet(test_path)
    logger.info("Loading model artifact: %s", model_path)
    model_artifact = joblib.load(model_path)

    logger.info("Predicting zoneRisk on test split...")
    zone_df = predict_zone_risk(test_df, model_artifact, target_col=target_col)
    if np.any((zone_df["zoneRisk"] < 0.0) | (zone_df["zoneRisk"] > 1.0)):
        raise ValueError("zoneRisk outside [0,1]")

    y_true = zone_df[target_col].to_numpy(dtype=int)
    y_prob = zone_df["zoneRisk"].to_numpy(dtype=float)
    base_rate = float(np.mean(y_true))
    report_rows = threshold_metrics(
        y_true,
        y_prob,
        thresholds,
        horizon_k=horizon_k,
        base_rate=base_rate,
    )

    report = {
        "timeframe": timeframe,
        "timeframe_minutes": _timeframe_to_minutes(timeframe),
        "horizon_k": horizon_k,
        "base_rate_test": base_rate,
        "baseline_definition": "lift_vs_base uses base positive rate on test set",
        "thresholds": report_rows,
    }

    out_zone_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    zone_df.to_parquet(out_zone_path, index=False)
    with open(out_report_path, "w") as file:
        json.dump(report, file, indent=2)

    logger.info("zoneRisk series saved: %s", out_zone_path)
    logger.info("Threshold report saved: %s", out_report_path)
    for row in report_rows:
        logger.info(
            "Threshold %.2f | flagged=%s | coverage=%.4f | hit_rate=%s | lift=%s | hit_within_k=%s",
            row["threshold"],
            row["flagged_bars"],
            row["coverage"],
            "null" if row["hit_rate"] is None else f"{row['hit_rate']:.4f}",
            "null" if row["lift_vs_base"] is None else f"{row['lift_vs_base']:.3f}",
            "null"
            if row["hit_within_k_rate"] is None
            else f"{row['hit_within_k_rate']:.4f}",
        )
    logger.info(
        "Scanner summary | rows=%s | base_rate=%.4f | thresholds=%s | elapsed=%.2fs",
        len(zone_df),
        base_rate,
        thresholds,
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()
