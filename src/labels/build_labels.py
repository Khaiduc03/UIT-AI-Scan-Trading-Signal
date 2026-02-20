import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Đọc cấu hình chung của project từ YAML (single source of truth).
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Tính future_range cho mỗi nến t theo cửa sổ tương lai [t+1 .. t+K].
# - Không dùng shift mơ hồ, dùng sliding_window_view để mapping index rõ ràng.
# - Kết quả giữ cùng độ dài raw_df, K dòng cuối sẽ là NaN vì không đủ dữ liệu tương lai.
def compute_future_range(raw_df: pd.DataFrame, horizon_k: int) -> pd.Series:
    if horizon_k <= 0:
        raise ValueError("horizon_k must be > 0")

    highs = raw_df["high"].to_numpy(dtype=float)
    lows = raw_df["low"].to_numpy(dtype=float)
    n = len(raw_df)

    future_range = np.full(n, np.nan, dtype=float)
    if n <= horizon_k:
        return pd.Series(future_range)

    high_windows = np.lib.stride_tricks.sliding_window_view(highs[1:], horizon_k)
    low_windows = np.lib.stride_tricks.sliding_window_view(lows[1:], horizon_k)

    valid_count = high_windows.shape[0]
    future_max_high = high_windows.max(axis=1)
    future_min_low = low_windows.min(axis=1)
    future_range[:valid_count] = future_max_high - future_min_low

    return pd.Series(future_range)


# Xây dựng bảng label StrongMove đã align theo timeline của features_core.
# Logic:
# 1) Tính future_range từ raw (dùng tương lai t+1..t+K)
# 2) Join theo time với atr14 tại thời điểm t (không leak feature)
# 3) strongmove_threshold = a * atr14(t)
# 4) StrongMove = 1 nếu future_range >= threshold, ngược lại 0
# 5) Drop các dòng thiếu future_range (thực tế là K dòng cuối)
def build_strongmove_labels(
    raw_df: pd.DataFrame,
    features_core_df: pd.DataFrame,
    horizon_k: int,
    strongmove_atr_mult: float,
) -> pd.DataFrame:
    raw_required = {"time", "high", "low"}
    core_required = {"time", "atr14"}
    if not raw_required.issubset(raw_df.columns):
        raise ValueError(f"raw_df must include columns: {sorted(raw_required)}")
    if not core_required.issubset(features_core_df.columns):
        raise ValueError(f"features_core_df must include columns: {sorted(core_required)}")

    future_range = compute_future_range(raw_df, horizon_k=horizon_k)
    raw_future = pd.DataFrame(
        {
            "time": raw_df["time"],
            "future_range": future_range,
        }
    )

    merged = features_core_df[["time", "atr14"]].merge(
        raw_future,
        on="time",
        how="inner",
        validate="one_to_one",
    )
    merged["strongmove_threshold"] = strongmove_atr_mult * merged["atr14"]
    merged = merged.dropna(subset=["future_range", "atr14"]).copy()
    merged["StrongMove"] = (merged["future_range"] >= merged["strongmove_threshold"]).astype("int8")

    return merged[["time", "atr14", "future_range", "strongmove_threshold", "StrongMove"]]


# Tạo report phân phối nhãn để review độ mất cân bằng lớp.
# Report này phục vụ acceptance criteria "label distribution is reported".
def build_distribution_report(
    labels_df: pd.DataFrame,
    horizon_k: int,
    strongmove_atr_mult: float,
    source_raw_path: str,
    source_features_path: str,
    output_labels_path: str,
) -> dict:
    total_rows = int(len(labels_df))
    positive_count = int(labels_df["StrongMove"].sum()) if total_rows else 0
    negative_count = total_rows - positive_count
    positive_rate = float(positive_count / total_rows) if total_rows else 0.0

    return {
        "total_rows": total_rows,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_rate,
        "horizon_k": int(horizon_k),
        "strongmove_atr_mult": float(strongmove_atr_mult),
        "source_raw_path": source_raw_path,
        "source_features_path": source_features_path,
        "output_labels_path": output_labels_path,
    }


# Entry point của Phase 3:
# - Đọc config + input files
# - Build StrongMove labels
# - Ghi parquet + JSON report
def main():
    cfg = load_config("configs/config.yaml")

    data_cfg = cfg.get("data", {})
    split_cfg = cfg.get("split", {})
    label_cfg = cfg.get("label", {})

    raw_path = data_cfg.get("output_csv", "artifacts/raw/BTCUSDT_15m.csv")
    processed_dir = Path(split_cfg.get("output_dir", "artifacts/processed"))
    features_core_path = processed_dir / "features_core.parquet"

    out_labels_path = Path(
        label_cfg.get("out_strongmove", processed_dir / "labels_strongmove.parquet")
    )
    out_report_path = Path(
        label_cfg.get("out_strongmove_report", "artifacts/reports/labels_strongmove_report.json")
    )

    horizon_k = int(label_cfg.get("horizon_k", 12))
    strongmove_atr_mult = float(label_cfg.get("strongmove_atr_mult", 2.5))

    if not Path(raw_path).exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    if not features_core_path.exists():
        raise FileNotFoundError(f"Core features not found: {features_core_path}")

    logger.info(f"Loading raw data: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    logger.info(f"Loading core features: {features_core_path}")
    features_core_df = pd.read_parquet(features_core_path)

    logger.info("Building StrongMove labels...")
    labels_df = build_strongmove_labels(
        raw_df=raw_df,
        features_core_df=features_core_df,
        horizon_k=horizon_k,
        strongmove_atr_mult=strongmove_atr_mult,
    )

    out_labels_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)

    labels_df.to_parquet(out_labels_path, index=False)
    report = build_distribution_report(
        labels_df=labels_df,
        horizon_k=horizon_k,
        strongmove_atr_mult=strongmove_atr_mult,
        source_raw_path=str(raw_path),
        source_features_path=str(features_core_path),
        output_labels_path=str(out_labels_path),
    )
    with open(out_report_path, "w") as file:
        json.dump(report, file, indent=2)

    logger.info(
        "StrongMove labels saved: rows=%s, positive_rate=%.6f, output=%s",
        report["total_rows"],
        report["positive_rate"],
        out_labels_path,
    )
    logger.info(f"Distribution report saved: {out_report_path}")


if __name__ == "__main__":
    main()
