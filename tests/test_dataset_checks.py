import pandas as pd

from src.evaluation.check_dataset_splits import _class_balance, _drift_report


def test_class_balance_basic():
    df = pd.DataFrame({"StrongMove": [1, 0, 1, 1, 0]})
    info = _class_balance(df, "StrongMove")
    assert info["rows"] == 5
    assert info["positive_count"] == 3
    assert info["negative_count"] == 2
    assert info["positive_rate"] == 0.6


def test_drift_report_has_rows():
    train = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [10.0, 10.0, 10.0, 10.0],
            "StrongMove": [0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "f1": [2.0, 3.0, 4.0, 5.0],
            "f2": [9.0, 9.0, 9.0, 9.0],
            "StrongMove": [1, 0, 1, 0],
        }
    )
    report = _drift_report(train, test, "StrongMove")
    assert report["feature_count_checked"] == 2
    assert len(report["top_mean_shift_features"]) >= 1
