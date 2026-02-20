import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.train_model1 import (
    LEAKAGE_COLS,
    eval_at_threshold,
    extract_features,
    select_threshold,
)


def _mock_split(n: int = 20) -> pd.DataFrame:
    times = (
        pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        .strftime("%Y-%m-%dT%H:%M:%SZ")
        .tolist()
    )
    x1 = np.linspace(0.0, 1.0, n)
    x2 = np.linspace(1.0, 2.0, n)
    y = np.array(([0, 1] * (n // 2)) + ([0] if n % 2 else []))
    return pd.DataFrame({"time": times, "f1": x1, "f2": x2, "StrongMove": y})


def test_extract_features_contract():
    train = _mock_split(20)
    val = _mock_split(10)
    test = _mock_split(10)
    X_train, X_val, X_test, y_train, y_val, y_test, cols = extract_features(
        train, val, test, "StrongMove"
    )
    assert cols == ["f1", "f2"]
    assert list(X_train.columns) == cols
    assert list(X_val.columns) == cols
    assert list(X_test.columns) == cols
    assert y_train.isin([0, 1]).all()
    assert y_val.isin([0, 1]).all()
    assert y_test.isin([0, 1]).all()


def test_select_threshold_policies():
    y_val = pd.Series([0, 0, 1, 1, 1, 0, 1, 0])
    p_val = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.4, 0.6, 0.3])

    cfg_fb = {"threshold_policy": "f_beta", "f_beta": 0.5, "threshold_grid_coarse": [0.3, 0.5, 0.7]}
    thr_fb, score_fb = select_threshold(y_val, p_val, cfg_fb)
    assert 0.0 <= thr_fb <= 1.0
    assert score_fb >= 0.0

    cfg_f1 = {"threshold_policy": "f1", "threshold_grid_coarse": [0.3, 0.5, 0.7]}
    thr_f1, score_f1 = select_threshold(y_val, p_val, cfg_f1)
    assert 0.0 <= thr_f1 <= 1.0
    assert score_f1 >= 0.0

    cfg_pr = {
        "threshold_policy": "max_precision_at_recall",
        "min_recall": 0.5,
        "threshold_grid_coarse": [0.3, 0.5, 0.7],
    }
    thr_pr, score_pr = select_threshold(y_val, p_val, cfg_pr)
    assert 0.0 <= thr_pr <= 1.0
    assert score_pr >= 0.0


def test_metrics_schema_and_confusion_ints():
    y = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
    p = np.array([0.1, 0.8, 0.3, 0.7, 0.9, 0.2, 0.4, 0.6])
    out = eval_at_threshold(y, p, threshold=0.5, bins=5)
    assert {
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
    }.issubset(out.keys())
    cm = out["confusion_matrix"]
    assert set(cm.keys()) == {"tn", "fp", "fn", "tp"}
    assert all(isinstance(cm[k], int) for k in cm)


def test_leakage_guard_raises():
    train = _mock_split(20).assign(future_range=0.0)
    val = _mock_split(10).assign(future_range=0.0)
    test = _mock_split(10).assign(future_range=0.0)
    with pytest.raises(ValueError, match="Leakage"):
        extract_features(train, val, test, "StrongMove")
    assert "future_range" in LEAKAGE_COLS


def test_artifact_load_predict_smoke(tmp_path: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train = _mock_split(20)
    val = _mock_split(10)
    test = _mock_split(10)
    X_train, _, X_test, y_train, _, _, cols = extract_features(train, val, test, "StrongMove")

    pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=200))])
    pipe.fit(X_train, y_train)
    artifact = {"pipeline": pipe, "feature_columns": cols}
    out_path = tmp_path / "model.pkl"
    joblib.dump(artifact, out_path)
    loaded = joblib.load(out_path)
    probs = loaded["pipeline"].predict_proba(X_test[loaded["feature_columns"]])[:, 1]
    assert np.all((0.0 <= probs) & (probs <= 1.0))


def test_integration_outputs_exist_if_trained():
    model_path = Path("artifacts/models/model1.pkl")
    metrics_path = Path("artifacts/reports/model1_metrics.json")
    if not (model_path.exists() and metrics_path.exists()):
        pytest.skip("Phase 5 artifacts not generated yet")
    data = json.load(open(metrics_path))
    assert "val" in data and "test" in data
