import pandas as pd
import pytest

from src.models.build_dataset import (
    build_dataset,
    build_split_metadata,
    split_time_dates,
    split_time_ratio,
    validate_splits,
)


def _sample_times(n: int = 10) -> list[str]:
    return (
        pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        .strftime("%Y-%m-%dT%H:%M:%SZ")
        .tolist()
    )


def test_merge_integrity_and_no_label_leakage_columns():
    times = _sample_times(6)
    core = pd.DataFrame({"time": times, "atr14": [1, 1, 1, 1, 1, 1], "ret1": [0, 1, 2, 3, 4, 5]})
    structure = pd.DataFrame({"time": times, "near_structure": [0.1] * 6})
    labels = pd.DataFrame(
        {
            "time": times,
            "atr14": [9] * 6,
            "future_range": [10] * 6,
            "strongmove_threshold": [2.5] * 6,
            "StrongMove": [0, 1, 0, 1, 0, 1],
        }
    )

    dataset = build_dataset(core, structure, labels)

    assert "StrongMove" in dataset.columns
    assert "future_range" not in dataset.columns
    assert "strongmove_threshold" not in dataset.columns
    assert "atr14_label" not in dataset.columns
    assert dataset["atr14"].tolist() == [1, 1, 1, 1, 1, 1]


def test_time_ratio_split_correctness():
    times = _sample_times(10)
    dataset = pd.DataFrame({"time": times, "f1": range(10), "StrongMove": [0, 1] * 5}).sort_values(
        "time"
    )

    train_df, val_df, test_df = split_time_ratio(dataset, 0.7, 0.2, 0.1)
    checks = validate_splits(dataset, train_df, val_df, test_df)

    assert len(train_df) == 7
    assert len(val_df) == 2
    assert len(test_df) == 1
    assert checks["is_disjoint_complete_partition"] is True
    assert checks["has_overlap"] is False


def test_time_dates_split_correctness():
    times = (
        pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
        .strftime("%Y-%m-%dT%H:%M:%SZ")
        .tolist()
    )
    dataset = pd.DataFrame({"time": times, "f1": range(8), "StrongMove": [0, 1] * 4}).sort_values(
        "time"
    )

    train_df, val_df, test_df = split_time_dates(
        dataset=dataset,
        train_end="2024-01-03",
        val_end="2024-01-06",
        test_end="2024-01-08",
    )
    checks = validate_splits(dataset, train_df, val_df, test_df)

    assert len(train_df) == 3
    assert len(val_df) == 3
    assert len(test_df) == 2
    assert checks["is_disjoint_complete_partition"] is True


def test_metadata_structure():
    times = _sample_times(8)
    dataset = pd.DataFrame({"time": times, "f1": range(8), "StrongMove": [0, 1] * 4}).sort_values(
        "time"
    )
    train_df, val_df, test_df = split_time_ratio(dataset, 0.5, 0.25, 0.25)

    metadata = build_split_metadata(
        "time_ratio",
        dataset,
        train_df,
        val_df,
        test_df,
        target_col="StrongMove",
    )

    assert metadata["method"] == "time_ratio"
    assert metadata["total_rows"] == 8
    assert metadata["target_col"] == "StrongMove"
    assert metadata["feature_count"] == 1
    assert set(metadata["splits"].keys()) == {"train", "val", "test"}
    assert set(metadata["integrity_checks"].keys()) == {
        "is_time_sorted",
        "has_overlap",
        "is_disjoint_complete_partition",
    }


def test_failure_modes():
    times = _sample_times(5)
    dataset = pd.DataFrame(
        {"time": times, "f1": range(5), "StrongMove": [0, 1, 0, 1, 0]}
    ).sort_values("time")

    with pytest.raises(ValueError, match="must sum to 1.0"):
        split_time_ratio(dataset, 0.7, 0.2, 0.2)

    dup = dataset.copy()
    dup.loc[1, "time"] = dup.loc[0, "time"]
    with pytest.raises(ValueError, match="overlap"):
        train_df = dataset.iloc[:2]
        val_df = dup.iloc[:2]
        test_df = dataset.iloc[2:]
        validate_splits(dataset, train_df, val_df, test_df)
