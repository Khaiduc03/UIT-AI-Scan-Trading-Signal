import pandas as pd
import pytest

from src.labels.build_labels_model2 import (
    LABEL_ID_MAPPING,
    build_distribution_report,
    build_model2_labels,
)


def test_future_window_correctness_and_tail_drop():
    raw_df = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC").astype(str),
            "high": [10, 12, 14, 13, 15, 11],
            "low": [9, 8, 7, 10, 9, 8],
            "close": [9.5, 11, 8, 12, 10, 9],
        }
    )
    core_df = pd.DataFrame(
        {
            "time": raw_df["time"],
            "atr14": [1, 1, 1, 1, 1, 1],
        }
    )

    labels_df, stats = build_model2_labels(
        raw_df=raw_df,
        features_core_df=core_df,
        horizon_k=2,
        atr_mult=1.0,
        neutral_rule="on_tie_or_below_threshold",
        mode="magnitude",
    )

    # t0 row: up=max(12,14)-9.5=4.5 ; down=9.5-min(8,7)=2.5
    assert labels_df.iloc[0]["up_move"] == pytest.approx(4.5)
    assert labels_df.iloc[0]["down_move"] == pytest.approx(2.5)
    # tail K rows removed
    assert len(labels_df) == 4
    assert stats["dropped_tail_rows"] == 2
    assert "both_hit" in labels_df.columns
    assert "is_tie" in labels_df.columns
    assert "neutral_reason" in labels_df.columns


def test_class_assignment_and_tie_to_neutral():
    raw_df = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC").astype(str),
            "high": [10, 14, 12, 12],
            "low": [10, 8, 12, 8],
            "close": [10, 10, 10, 10],
        }
    )
    core_df = pd.DataFrame(
        {
            "time": raw_df["time"],
            "atr14": [2, 2, 2, 2],
        }
    )

    labels_df, _ = build_model2_labels(
        raw_df=raw_df,
        features_core_df=core_df,
        horizon_k=1,
        atr_mult=1.0,
        neutral_rule="on_tie_or_below_threshold",
        mode="magnitude",
    )
    # row0: up=4, down=2 => long
    assert labels_df.iloc[0]["signal"] == "long"

    # build a tie-only case
    raw_tie = pd.DataFrame(
        {
            "time": pd.date_range("2025-02-01", periods=3, freq="15min", tz="UTC").astype(str),
            "high": [10, 13, 10],
            "low": [10, 7, 10],
            "close": [10, 10, 10],
        }
    )
    core_tie = pd.DataFrame({"time": raw_tie["time"], "atr14": [2, 2, 2]})
    labels_tie, _ = build_model2_labels(
        raw_df=raw_tie,
        features_core_df=core_tie,
        horizon_k=1,
        atr_mult=1.0,
        neutral_rule="on_tie_or_below_threshold",
        mode="magnitude",
    )
    assert labels_tie.iloc[0]["signal"] == "neutral"
    assert labels_tie.iloc[0]["label_id"] == LABEL_ID_MAPPING["neutral"]
    assert bool(labels_tie.iloc[0]["is_tie"]) is True
    assert labels_tie.iloc[0]["neutral_reason"] == "tie"


def test_atr_invalid_rows_dropped_and_counted():
    raw_df = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC").astype(str),
            "high": [10, 11, 12, 13, 14],
            "low": [9, 8, 7, 6, 5],
            "close": [9.5, 9.6, 9.7, 9.8, 9.9],
        }
    )
    core_df = pd.DataFrame(
        {
            "time": raw_df["time"],
            "atr14": [1, 0, -1, None, 1],
        }
    )
    labels_df, stats = build_model2_labels(
        raw_df=raw_df,
        features_core_df=core_df,
        horizon_k=1,
        atr_mult=1.0,
        neutral_rule="on_tie_or_below_threshold",
        mode="magnitude",
    )
    assert stats["dropped_due_to_atr_rows"] == 3
    assert len(labels_df) == 1


def test_join_integrity_error_on_duplicate_time():
    raw_df = pd.DataFrame(
        {
            "time": ["2025-01-01T00:00:00Z", "2025-01-01T00:15:00Z"],
            "high": [10, 11],
            "low": [9, 8],
            "close": [9.5, 9.6],
        }
    )
    core_df = pd.DataFrame(
        {
            "time": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "atr14": [1, 1],
        }
    )
    with pytest.raises(ValueError, match="duplicated timestamps|one-to-one"):
        build_model2_labels(
            raw_df=raw_df,
            features_core_df=core_df,
            horizon_k=1,
            atr_mult=1.0,
            neutral_rule="on_tie_or_below_threshold",
            mode="magnitude",
        )


def test_time_integrity_error_on_non_monotonic_time():
    raw_df = pd.DataFrame(
        {
            "time": ["2025-01-01T00:15:00Z", "2025-01-01T00:00:00Z", "2025-01-01T00:30:00Z"],
            "high": [10, 11, 12],
            "low": [9, 8, 7],
            "close": [9.5, 9.6, 9.7],
        }
    )
    core_df = pd.DataFrame(
        {
            "time": ["2025-01-01T00:00:00Z", "2025-01-01T00:15:00Z", "2025-01-01T00:30:00Z"],
            "atr14": [1, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="monotonic increasing"):
        build_model2_labels(
            raw_df=raw_df,
            features_core_df=core_df,
            horizon_k=1,
            atr_mult=1.0,
            neutral_rule="on_tie_or_below_threshold",
            mode="magnitude",
        )


def test_report_schema_and_mapping():
    labels_df = pd.DataFrame(
        {
            "time": ["2025-01-01T00:00:00Z", "2025-01-01T00:15:00Z", "2025-01-01T00:30:00Z"],
            "atr14": [1.0, 1.0, 1.0],
            "up_move": [2.0, 0.5, 0.2],
            "down_move": [0.1, 1.5, 0.2],
            "threshold": [1.0, 1.0, 1.0],
            "both_hit": [False, False, True],
            "is_tie": [False, False, True],
            "neutral_reason": [None, None, "tie"],
            "signal": ["long", "short", "neutral"],
            "label_id": [1, 2, 0],
        }
    )
    stats = {
        "dropped_tail_rows": 2,
        "dropped_due_to_atr_rows": 1,
        "dropped_due_to_join_rows": 0,
        "ambiguous_both_hit_rows": 1,
        "ambiguous_both_hit_tie_rows": 1,
        "neutral_breakdown_counts": {
            "below_threshold": 0,
            "both_hit": 0,
            "tie": 1,
            "winner_but_below_thr": 0,
        },
    }
    report = build_distribution_report(
        labels_df=labels_df,
        anchor="bar",
        horizon_k=12,
        atr_mult=1.5,
        neutral_rule="on_tie_or_below_threshold",
        mode="magnitude",
        output_labels_path="artifacts/processed/labels_model2_bar.parquet",
        input_paths={
            "raw_path": "artifacts/raw/BTCUSDT_15m.csv",
            "features_core_path": "artifacts/processed/features_core.parquet",
        },
        stats=stats,
    )
    assert report["label_id_mapping"] == LABEL_ID_MAPPING
    assert report["class_counts"] == {"long": 1, "short": 1, "neutral": 1}
    assert report["dropped_tail_rows"] == 2
    assert report["ambiguous_both_hit_rate"] == pytest.approx(1 / 3)
    assert report["neutral_breakdown_counts"]["tie"] == 1
    assert report["ambiguous_both_hit_counts"]["tie"] == 1
