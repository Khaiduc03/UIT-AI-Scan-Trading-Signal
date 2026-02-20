import numpy as np
import pandas as pd

from src.labels.build_labels import (
    build_distribution_report,
    build_strongmove_labels,
    compute_future_range,
)


def _raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [f"2024-01-01T00:{i:02d}:00Z" for i in range(6)],
            "high": [10, 11, 12, 13, 14, 15],
            "low": [5, 6, 7, 8, 9, 10],
            "close": [7, 8, 9, 10, 11, 12],
        }
    )


def test_future_range_window_correctness():
    raw = _raw_sample()
    # t=0 -> max(11,12)-min(6,7)=6 ; t=1 -> max(12,13)-min(7,8)=6
    got = compute_future_range(raw, horizon_k=2)
    expected = [6.0, 6.0, 6.0, 6.0, np.nan, np.nan]
    assert np.allclose(got.iloc[:4], expected[:4])
    assert got.iloc[4:].isna().all()


def test_tail_drop_and_alignment():
    raw = _raw_sample()
    core = pd.DataFrame(
        {
            "time": raw["time"],
            "atr14": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    labels = build_strongmove_labels(raw, core, horizon_k=2, strongmove_atr_mult=2.5)
    # last K=2 rows must be dropped
    assert len(labels) == len(raw) - 2
    assert labels["time"].is_unique
    assert set(labels["time"]).issubset(set(core["time"]))


def test_threshold_classification():
    raw = _raw_sample()
    core = pd.DataFrame(
        {
            "time": raw["time"],
            "atr14": [1.0, 3.0, 1.0, 3.0, 1.0, 3.0],
        }
    )
    labels = build_strongmove_labels(raw, core, horizon_k=2, strongmove_atr_mult=2.5)
    # future_range on valid rows is 6.0
    # thresholds: 2.5, 7.5, 2.5, 7.5 -> StrongMove: 1,0,1,0
    assert labels["StrongMove"].tolist() == [1, 0, 1, 0]
    assert labels["StrongMove"].dtype == "int8"


def test_distribution_report_fields():
    labels = pd.DataFrame({"StrongMove": pd.Series([1, 0, 1, 0], dtype="int8")})
    report = build_distribution_report(
        labels_df=labels,
        horizon_k=12,
        strongmove_atr_mult=2.5,
        source_raw_path="raw.csv",
        source_features_path="features.parquet",
        output_labels_path="labels.parquet",
    )
    assert report["total_rows"] == 4
    assert report["positive_count"] == 2
    assert report["negative_count"] == 2
    assert report["positive_rate"] == 0.5
    assert report["horizon_k"] == 12
