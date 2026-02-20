import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Đọc cấu hình chung của project (single source of truth).
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Guard helper: đảm bảo key thời gian là duy nhất trước khi merge one-to-one.
def _assert_time_unique(df: pd.DataFrame, name: str) -> None:
    if not df["time"].is_unique:
        raise ValueError(f"{name} has duplicate `time` values")


def build_dataset(
    features_core: pd.DataFrame,
    features_structure: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    # Mục tiêu:
    # 1) Ghép core + structure thành features_all theo time.
    # 2) Ghép features_all với labels StrongMove.
    # 3) Loại các cột label-helper để tránh leakage lúc train.
    # 4) Sort thời gian tăng dần và ép StrongMove về int8.
    required_core = {"time"}
    required_structure = {"time"}
    required_labels = {"time", "StrongMove"}
    if not required_core.issubset(features_core.columns):
        raise ValueError("features_core must contain `time`")
    if not required_structure.issubset(features_structure.columns):
        raise ValueError("features_structure must contain `time`")
    if not required_labels.issubset(labels.columns):
        raise ValueError("labels must contain `time` and `StrongMove`")

    _assert_time_unique(features_core, "features_core")
    _assert_time_unique(features_structure, "features_structure")
    _assert_time_unique(labels, "labels")

    features_all = features_core.merge(
        features_structure,
        on="time",
        how="inner",
        validate="one_to_one",
    )
    labels_used = labels.drop(
        columns=["future_range", "strongmove_threshold"],
        errors="ignore",
    )
    dataset = features_all.merge(
        labels_used,
        on="time",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_label"),
    )

    # Ưu tiên atr14 từ features, không dùng bản duplicate đi theo labels.
    dataset = dataset.drop(columns=["atr14_label"], errors="ignore")

    if "StrongMove" not in dataset.columns:
        raise ValueError("Merged dataset does not include StrongMove")

    dataset = dataset.sort_values("time").reset_index(drop=True)
    dataset["StrongMove"] = dataset["StrongMove"].astype("int8")

    return dataset


def split_time_ratio(
    dataset: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Chia dữ liệu theo tỉ lệ thời gian, tuyệt đối không shuffle.
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum}")

    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = dataset.iloc[:train_end].copy()
    val_df = dataset.iloc[train_end:val_end].copy()
    test_df = dataset.iloc[val_end:].copy()
    return train_df, val_df, test_df


def split_time_dates(
    dataset: pd.DataFrame,
    train_end: str,
    val_end: str,
    test_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Chia theo mốc ngày:
    # - train: time <= train_end
    # - val: train_end < time <= val_end
    # - test: time > val_end (và <= test_end nếu có)
    if not train_end or not val_end:
        raise ValueError("time_dates mode requires split.train_end and split.val_end")

    time_dt = pd.to_datetime(dataset["time"], utc=True)
    train_end_dt = pd.to_datetime(train_end, utc=True)
    val_end_dt = pd.to_datetime(val_end, utc=True)
    test_end_dt = pd.to_datetime(test_end, utc=True) if test_end else None

    train_mask = time_dt <= train_end_dt
    val_mask = (time_dt > train_end_dt) & (time_dt <= val_end_dt)
    test_mask = time_dt > val_end_dt
    if test_end_dt is not None:
        test_mask = test_mask & (time_dt <= test_end_dt)

    train_df = dataset.loc[train_mask].copy()
    val_df = dataset.loc[val_mask].copy()
    test_df = dataset.loc[test_mask].copy()
    return train_df, val_df, test_df


def validate_splits(
    dataset: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    # Kiểm tra integrity của split:
    # - dataset không rỗng, mỗi split không rỗng
    # - sorted theo time
    # - không overlap giữa train/val/test
    # - tổng số dòng split khớp dataset
    # - thứ tự thời gian strict: train < val < test
    if dataset.empty:
        raise ValueError("Dataset is empty")
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more splits are empty")

    dataset_sorted = dataset["time"].is_monotonic_increasing
    if not dataset_sorted:
        raise ValueError("Dataset must be sorted by time")

    train_times = set(train_df["time"])
    val_times = set(val_df["time"])
    test_times = set(test_df["time"])
    has_overlap = bool(
        (train_times & val_times) or (train_times & test_times) or (val_times & test_times)
    )
    if has_overlap:
        raise ValueError("Split overlap detected")

    combined_count = len(train_df) + len(val_df) + len(test_df)
    complete_partition = combined_count == len(dataset)
    if not complete_partition:
        raise ValueError("Split rows do not match dataset rows")

    if not (
        train_df["time"].max() < val_df["time"].min()
        and val_df["time"].max() < test_df["time"].min()
    ):
        raise ValueError("Splits are not strictly ordered by time")

    return {
        "is_time_sorted": dataset_sorted,
        "has_overlap": has_overlap,
        "is_disjoint_complete_partition": complete_partition,
    }


def _split_info(df: pd.DataFrame, total_rows: int) -> dict:
    # Thông tin tóm tắt cho mỗi split để ghi metadata.
    return {
        "rows": int(len(df)),
        "start_time": str(df["time"].iloc[0]),
        "end_time": str(df["time"].iloc[-1]),
        "ratio_actual": float(len(df) / total_rows),
    }


def build_split_metadata(
    method: str,
    dataset: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "StrongMove",
) -> dict:
    # Tạo metadata splits.json phục vụ trace và kiểm tra pipeline.
    integrity = validate_splits(dataset, train_df, val_df, test_df)
    total_rows = len(dataset)
    feature_cols = [c for c in dataset.columns if c not in {"time", target_col}]

    return {
        "method": method,
        "total_rows": int(total_rows),
        "target_col": target_col,
        "feature_count": int(len(feature_cols)),
        "splits": {
            "train": _split_info(train_df, total_rows),
            "val": _split_info(val_df, total_rows),
            "test": _split_info(test_df, total_rows),
        },
        "integrity_checks": integrity,
    }


def main():
    # 1) Đọc config + resolve input/output paths.
    cfg = load_config("configs/config.yaml")

    split_cfg = cfg.get("split", {})
    processed_dir = Path(split_cfg.get("output_dir", "artifacts/processed"))
    method = split_cfg.get("method", "time_ratio")

    features_core_path = processed_dir / "features_core.parquet"
    features_structure_path = processed_dir / "features_structure.parquet"
    labels_path = Path(
        cfg.get("label", {}).get("out_strongmove", processed_dir / "labels_strongmove.parquet")
    )

    out_dataset = Path(split_cfg.get("out_dataset", processed_dir / "dataset_model1.parquet"))
    out_train = Path(split_cfg.get("out_train", processed_dir / "train.parquet"))
    out_val = Path(split_cfg.get("out_val", processed_dir / "val.parquet"))
    out_test = Path(split_cfg.get("out_test", processed_dir / "test.parquet"))
    out_splits = Path(split_cfg.get("out_splits", processed_dir / "splits.json"))

    # 2) Fail-fast nếu thiếu input artifacts cần thiết.
    for p in [features_core_path, features_structure_path, labels_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    # 3) Load dữ liệu và build dataset hợp nhất.
    features_core = pd.read_parquet(features_core_path)
    features_structure = pd.read_parquet(features_structure_path)
    labels = pd.read_parquet(labels_path)

    dataset = build_dataset(features_core, features_structure, labels)

    # 4) Split theo method trong config (time_ratio hoặc time_dates).
    if method == "time_ratio":
        train_df, val_df, test_df = split_time_ratio(
            dataset=dataset,
            train_ratio=float(split_cfg.get("train_ratio", 0.7)),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        )
    elif method == "time_dates":
        train_df, val_df, test_df = split_time_dates(
            dataset=dataset,
            train_end=split_cfg.get("train_end"),
            val_end=split_cfg.get("val_end"),
            test_end=split_cfg.get("test_end"),
        )
    else:
        raise ValueError(f"Unsupported split.method: {method}")

    # 5) Build metadata và ghi tất cả artifacts Phase 4.
    metadata = build_split_metadata(
        method,
        dataset,
        train_df,
        val_df,
        test_df,
        target_col="StrongMove",
    )

    for out_path in [out_dataset, out_train, out_val, out_test, out_splits]:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(out_dataset, index=False)
    train_df.to_parquet(out_train, index=False)
    val_df.to_parquet(out_val, index=False)
    test_df.to_parquet(out_test, index=False)
    with open(out_splits, "w") as file:
        json.dump(metadata, file, indent=2)

    logger.info(
        "Dataset built: total=%s train=%s val=%s test=%s method=%s",
        len(dataset),
        len(train_df),
        len(val_df),
        len(test_df),
        method,
    )


if __name__ == "__main__":
    main()
