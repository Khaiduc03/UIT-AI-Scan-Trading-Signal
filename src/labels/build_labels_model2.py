import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LABEL_ID_MAPPING = {"neutral": 0, "long": 1, "short": 2}
ALLOWED_NEUTRAL_RULES = {"on_tie_or_below_threshold"}
ALLOWED_LABEL_MODES = {"magnitude"}
TIE_EPS = 1e-12


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def normalize_time_utc(df: pd.DataFrame, col: str = "time") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True, errors="raise")
    return out


def validate_time_integrity(df: pd.DataFrame, col: str, source_name: str) -> None:
    if df[col].isna().any():
        raise ValueError(f"{source_name}.{col} contains NaT after parsing")
    if df[col].duplicated().any():
        raise ValueError(f"{source_name}.{col} contains duplicated timestamps")
    if not df[col].is_monotonic_increasing:
        raise ValueError(f"{source_name}.{col} must be monotonic increasing")


def _future_max_min(series: pd.Series, horizon_k: int, fn: str) -> pd.Series:
    shifted = pd.concat([series.shift(-i) for i in range(1, horizon_k + 1)], axis=1)
    valid_counts = shifted.notna().sum(axis=1)
    if fn == "max":
        values = shifted.max(axis=1)
    else:
        values = shifted.min(axis=1)
    values[valid_counts < horizon_k] = np.nan
    return values


def compute_future_moves(raw_df: pd.DataFrame, horizon_k: int) -> pd.DataFrame:
    if horizon_k <= 0:
        raise ValueError("label2.horizon_k must be > 0")

    future_max_high = _future_max_min(raw_df["high"], horizon_k, fn="max")
    future_min_low = _future_max_min(raw_df["low"], horizon_k, fn="min")
    up_move = future_max_high - raw_df["close"]
    down_move = raw_df["close"] - future_min_low
    return pd.DataFrame(
        {
            "time": raw_df["time"],
            "up_move": up_move,
            "down_move": down_move,
        }
    )


def build_model2_labels(
    raw_df: pd.DataFrame,
    features_core_df: pd.DataFrame,
    horizon_k: int,
    atr_mult: float,
    neutral_rule: str,
    mode: str,
) -> tuple[pd.DataFrame, dict]:
    if neutral_rule not in ALLOWED_NEUTRAL_RULES:
        allowed = sorted(ALLOWED_NEUTRAL_RULES)
        raise ValueError(
            f"Unsupported label2.neutral_rule={neutral_rule!r}, allowed={allowed}"
        )
    if mode not in ALLOWED_LABEL_MODES:
        raise ValueError(f"Unsupported label2.mode={mode!r}, allowed={sorted(ALLOWED_LABEL_MODES)}")

    raw_required = {"time", "high", "low", "close"}
    core_required = {"time", "atr14"}
    if not raw_required.issubset(raw_df.columns):
        raise ValueError(f"raw_df must include columns: {sorted(raw_required)}")
    if not core_required.issubset(features_core_df.columns):
        raise ValueError(f"features_core_df must include columns: {sorted(core_required)}")

    raw_df = normalize_time_utc(raw_df, "time")
    features_core_df = normalize_time_utc(features_core_df, "time")
    validate_time_integrity(raw_df, "time", "raw_df")
    validate_time_integrity(features_core_df, "time", "features_core_df")

    raw_moves = compute_future_moves(raw_df, horizon_k=horizon_k)
    try:
        merged = features_core_df[["time", "atr14"]].merge(
            raw_moves,
            on="time",
            how="inner",
            validate="one_to_one",
        )
    except Exception as exc:  # pandas MergeError type changed by versions
        raise ValueError(f"Join failed (time one-to-one expected): {exc}") from exc

    dropped_due_to_join_rows = max(0, len(features_core_df) - len(merged))
    tail_mask = merged["up_move"].isna() | merged["down_move"].isna()
    dropped_tail_rows = int(tail_mask.sum())
    merged = merged.loc[~tail_mask].copy()

    invalid_atr_mask = merged["atr14"].isna() | (merged["atr14"] <= 0)
    dropped_due_to_atr_rows = int(invalid_atr_mask.sum())
    merged = merged.loc[~invalid_atr_mask].copy()

    merged["threshold"] = float(atr_mult) * merged["atr14"]
    merged["signal"] = "neutral"

    long_mask = (merged["up_move"] >= merged["threshold"]) & (
        merged["up_move"] > merged["down_move"]
    )
    short_mask = (merged["down_move"] >= merged["threshold"]) & (
        merged["down_move"] > merged["up_move"]
    )
    both_hit_mask = (merged["up_move"] >= merged["threshold"]) & (
        merged["down_move"] >= merged["threshold"]
    )
    is_tie_mask = (merged["up_move"] - merged["down_move"]).abs() <= TIE_EPS

    merged.loc[long_mask, "signal"] = "long"
    merged.loc[short_mask, "signal"] = "short"
    merged["both_hit"] = both_hit_mask.astype(bool)
    merged["is_tie"] = is_tie_mask.astype(bool)
    merged["neutral_reason"] = pd.Series([None] * len(merged), index=merged.index, dtype="object")

    neutral_mask = merged["signal"] == "neutral"
    merged.loc[neutral_mask & both_hit_mask, "neutral_reason"] = "both_hit"
    merged.loc[neutral_mask & is_tie_mask, "neutral_reason"] = "tie"
    winner_but_below_thr_mask = (
        neutral_mask
        & (~both_hit_mask)
        & (~is_tie_mask)
        & (merged["up_move"] < merged["threshold"])
        & (merged["down_move"] < merged["threshold"])
    )
    merged.loc[winner_but_below_thr_mask, "neutral_reason"] = "winner_but_below_thr"
    merged.loc[neutral_mask & merged["neutral_reason"].isna(), "neutral_reason"] = "below_threshold"

    merged["label_id"] = merged["signal"].map(LABEL_ID_MAPPING).astype("int8")
    merged["time"] = merged["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    neutral_breakdown_counts = (
        merged.loc[neutral_mask, "neutral_reason"].value_counts(dropna=False).to_dict()
    )
    ambiguous_both_hit_tie_rows = int((both_hit_mask & is_tie_mask).sum())

    stats = {
        "dropped_tail_rows": dropped_tail_rows,
        "dropped_due_to_atr_rows": dropped_due_to_atr_rows,
        "dropped_due_to_join_rows": dropped_due_to_join_rows,
        "ambiguous_both_hit_rows": int(both_hit_mask.sum()),
        "ambiguous_both_hit_tie_rows": ambiguous_both_hit_tie_rows,
        "neutral_breakdown_counts": {
            "below_threshold": int(neutral_breakdown_counts.get("below_threshold", 0)),
            "both_hit": int(neutral_breakdown_counts.get("both_hit", 0)),
            "tie": int(neutral_breakdown_counts.get("tie", 0)),
            "winner_but_below_thr": int(neutral_breakdown_counts.get("winner_but_below_thr", 0)),
        },
    }
    return merged[
        [
            "time",
            "atr14",
            "up_move",
            "down_move",
            "threshold",
            "both_hit",
            "is_tie",
            "neutral_reason",
            "signal",
            "label_id",
        ]
    ], stats


def build_distribution_report(
    labels_df: pd.DataFrame,
    *,
    anchor: str,
    horizon_k: int,
    atr_mult: float,
    neutral_rule: str,
    mode: str,
    output_labels_path: str,
    input_paths: dict,
    stats: dict,
) -> dict:
    total_rows = int(len(labels_df))
    class_counts = {
        "long": int((labels_df["signal"] == "long").sum()),
        "short": int((labels_df["signal"] == "short").sum()),
        "neutral": int((labels_df["signal"] == "neutral").sum()),
    }
    class_rates = {
        k: (float(v) / total_rows if total_rows else 0.0) for k, v in class_counts.items()
    }
    ambiguous_both_hit_rows = int(stats["ambiguous_both_hit_rows"])
    ambiguous_both_hit_rate = (
        float(ambiguous_both_hit_rows / total_rows) if total_rows else 0.0
    )
    ambiguous_both_hit_tie_rows = int(stats["ambiguous_both_hit_tie_rows"])
    ambiguous_both_hit_counts = {
        "total": ambiguous_both_hit_rows,
        "tie": ambiguous_both_hit_tie_rows,
        "non_tie": ambiguous_both_hit_rows - ambiguous_both_hit_tie_rows,
    }

    return {
        "total_rows": total_rows,
        "class_counts": class_counts,
        "class_rates": class_rates,
        "label_id_mapping": LABEL_ID_MAPPING,
        "anchor": anchor,
        "horizon_k": int(horizon_k),
        "atr_mult": float(atr_mult),
        "neutral_rule": neutral_rule,
        "mode": mode,
        "dropped_tail_rows": int(stats["dropped_tail_rows"]),
        "dropped_due_to_atr_rows": int(stats["dropped_due_to_atr_rows"]),
        "dropped_due_to_join_rows": int(stats["dropped_due_to_join_rows"]),
        "ambiguous_both_hit_rows": ambiguous_both_hit_rows,
        "ambiguous_both_hit_rate": ambiguous_both_hit_rate,
        "neutral_breakdown_counts": stats["neutral_breakdown_counts"],
        "ambiguous_both_hit_counts": ambiguous_both_hit_counts,
        "config_snapshot": {
            "label2": {
                "anchor": anchor,
                "horizon_k": int(horizon_k),
                "atr_mult": float(atr_mult),
                "neutral_rule": neutral_rule,
                "mode": mode,
            },
            "inputs": input_paths,
        },
        "time_range": {
            "start_time": labels_df["time"].iloc[0] if total_rows else None,
            "end_time": labels_df["time"].iloc[-1] if total_rows else None,
        },
        "output_labels_path": output_labels_path,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def run(config_path: str = "configs/config.yaml") -> tuple[Path, Path]:
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})
    label2_cfg = cfg.get("label2", {})

    raw_path = Path(data_cfg.get("output_csv", "artifacts/raw/BTCUSDT_15m.csv"))
    processed_dir = Path(split_cfg.get("output_dir", "artifacts/processed"))
    features_core_path = processed_dir / "features_core.parquet"

    out_labels_path = Path(
        label2_cfg.get("out_labels", processed_dir / "labels_model2_bar.parquet")
    )
    out_report_path = Path(
        label2_cfg.get("out_report", "artifacts/reports/model2_label_distribution.json")
    )

    anchor = str(label2_cfg.get("anchor", "bar"))
    horizon_k = int(label2_cfg.get("horizon_k", 12))
    atr_mult = float(label2_cfg.get("atr_mult", 1.5))
    neutral_rule = str(label2_cfg.get("neutral_rule", "on_tie_or_below_threshold"))
    mode = str(label2_cfg.get("mode", "magnitude"))

    if anchor != "bar":
        raise ValueError(f"Phase 9 expects label2.anchor='bar', got {anchor!r}")
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    if not features_core_path.exists():
        raise FileNotFoundError(f"Core features not found: {features_core_path}")

    logger.info("Loading raw data: %s", raw_path)
    raw_df = pd.read_csv(raw_path)
    logger.info("Loading core features: %s", features_core_path)
    features_core_df = pd.read_parquet(features_core_path)

    labels_df, stats = build_model2_labels(
        raw_df=raw_df,
        features_core_df=features_core_df,
        horizon_k=horizon_k,
        atr_mult=atr_mult,
        neutral_rule=neutral_rule,
        mode=mode,
    )

    out_labels_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(out_labels_path, index=False)

    report = build_distribution_report(
        labels_df,
        anchor=anchor,
        horizon_k=horizon_k,
        atr_mult=atr_mult,
        neutral_rule=neutral_rule,
        mode=mode,
        output_labels_path=str(out_labels_path),
        input_paths={
            "raw_path": str(raw_path),
            "features_core_path": str(features_core_path),
        },
        stats=stats,
    )
    with open(out_report_path, "w") as file:
        json.dump(report, file, indent=2)

    logger.info(
        "Model2 labels saved: rows=%s long=%.4f short=%.4f neutral=%.4f output=%s",
        report["total_rows"],
        report["class_rates"]["long"],
        report["class_rates"]["short"],
        report["class_rates"]["neutral"],
        out_labels_path,
    )
    logger.info(
        (
            "Model2 label report saved: %s | dropped_tail=%s dropped_atr=%s "
            "dropped_join=%s ambiguous=%.4f"
        ),
        out_report_path,
        report["dropped_tail_rows"],
        report["dropped_due_to_atr_rows"],
        report["dropped_due_to_join_rows"],
        report["ambiguous_both_hit_rate"],
    )
    return out_labels_path, out_report_path


def main() -> None:
    start_time = perf_counter()
    labels_path, report_path = run()
    logger.info(
        "Phase 9 completed | labels=%s report=%s elapsed=%.2fs",
        labels_path,
        report_path,
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()
