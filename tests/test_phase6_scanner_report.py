import numpy as np

from src.evaluation.run_scanner_report import _hit_within_k_rate, threshold_metrics


def test_threshold_metrics_use_gte_and_counts():
    y_true = np.array([0, 1, 1, 0, 1], dtype=int)
    y_prob = np.array([0.5, 0.5, 0.8, 0.4, 0.5], dtype=float)
    # With >=0.5, flagged indexes = 0,1,2,4 => flagged_bars=4
    out = threshold_metrics(
        y_true,
        y_prob,
        thresholds=[0.5],
        horizon_k=2,
        base_rate=float(y_true.mean()),
    )
    row = out[0]
    assert row["flagged_bars"] == 4
    assert row["tp"] == 3
    assert row["fp"] == 1
    assert row["tn"] == 1
    assert row["fn"] == 0
    assert row["precision"] == row["hit_rate"]


def test_hit_within_k_rate_uses_t_plus_1_to_k():
    # y_true by index: 0 1 2 3 4 5
    # values          0 0 1 0 1 0
    y = np.array([0, 0, 1, 0, 1, 0], dtype=int)
    flags = np.array([True, True, False, True, False, False])
    # K=2
    # t=0 -> look [1,2] has 1 => hit
    # t=1 -> look [2,3] has 1 => hit
    # t=3 -> look [4,5] has 1 => hit
    rate = _hit_within_k_rate(y, flags, horizon_k=2)
    assert rate == 1.0


def test_null_safe_no_flagged_bars():
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_prob = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    out = threshold_metrics(
        y_true,
        y_prob,
        thresholds=[0.9],
        horizon_k=2,
        base_rate=float(y_true.mean()),
    )
    row = out[0]
    assert row["flagged_bars"] == 0
    assert row["hit_rate"] is None
    assert row["lift_vs_base"] is None
    assert row["precision"] is None
    assert row["recall"] is None
    assert row["f1"] is None
    assert row["hit_within_k_rate"] is None
    assert "note" in row
