import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LEAKAGE_COLS = {"future_range", "strongmove_threshold"}
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def _read_splits(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cfg = cfg.get("split", {})
    processed_dir = Path(split_cfg.get("output_dir", "artifacts/processed"))
    train_path = Path(split_cfg.get("out_train", processed_dir / "train.parquet"))
    val_path = Path(split_cfg.get("out_val", processed_dir / "val.parquet"))
    test_path = Path(split_cfg.get("out_test", processed_dir / "test.parquet"))
    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
    logger.info("Loading splits: train=%s val=%s test=%s", train_path, val_path, test_path)
    return pd.read_parquet(train_path), pd.read_parquet(val_path), pd.read_parquet(test_path)


def _validate_target(y: pd.Series, split_name: str) -> None:
    if y.empty:
        raise ValueError(f"{split_name} target is empty")
    uniq = set(y.unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"{split_name} target must be binary 0/1, got {sorted(uniq)}")
    if len(uniq) < 2:
        raise ValueError(f"{split_name} target is degenerate (single class)")


def extract_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if df.empty:
            raise ValueError(f"{name} split is empty")
        if target_col not in df.columns:
            raise ValueError(f"{name} split missing target column: {target_col}")

    candidate_cols = [c for c in train_df.columns if c not in {"time", target_col}]
    if any(c in LEAKAGE_COLS for c in candidate_cols):
        found = sorted(list(set(candidate_cols).intersection(LEAKAGE_COLS)))
        raise ValueError(f"Leakage columns found in features: {found}")

    # Keep only numeric features and enforce exact order across splits.
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found")

    expected_cols = set(numeric_cols + ["time", target_col])
    for name, df in [("val", val_df), ("test", test_df)]:
        if not set(numeric_cols).issubset(df.columns):
            raise ValueError(f"{name} split does not contain all feature columns")
        if set(df.columns) != expected_cols:
            raise ValueError(f"{name} split columns mismatch with train split")

    X_train = train_df[numeric_cols].copy()
    X_val = val_df[numeric_cols].copy()
    X_test = test_df[numeric_cols].copy()
    y_train = train_df[target_col].astype(int).copy()
    y_val = val_df[target_col].astype(int).copy()
    y_test = test_df[target_col].astype(int).copy()

    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} features contain NaN or inf")
    _validate_target(y_train, "train")
    _validate_target(y_val, "val")
    _validate_target(y_test, "test")

    return X_train, X_val, X_test, y_train, y_val, y_test, numeric_cols


def _threshold_scores(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: list[float],
    policy: str,
    f_beta: float,
    min_recall: float,
) -> tuple[float, float]:
    best_thr = None
    best_score = -math.inf
    best_precision = -math.inf

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        if policy == "f_beta":
            score = fbeta_score(y_true, y_pred, beta=f_beta, zero_division=0)
        elif policy == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif policy == "max_precision_at_recall":
            score = p if r >= min_recall else -math.inf
        else:
            raise ValueError(f"Unsupported threshold policy: {policy}")

        if (
            (score > best_score)
            or (score == best_score and p > best_precision)
            or (
                score == best_score and p == best_precision and (best_thr is None or thr > best_thr)
            )
        ):
            best_score = score
            best_precision = p
            best_thr = thr

    if best_thr is None or not np.isfinite(best_score):
        raise ValueError(f"No feasible threshold found for policy={policy}")
    return float(best_thr), float(best_score)


def select_threshold(
    y_val: pd.Series,
    y_prob_val: np.ndarray,
    model_cfg: dict,
) -> tuple[float, float]:
    policy = model_cfg.get("threshold_policy", "f_beta")
    f_beta = float(model_cfg.get("f_beta", 0.5))
    min_recall = float(model_cfg.get("min_recall", 0.60))
    coarse = list(
        model_cfg.get(
            "threshold_grid_coarse",
            [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)],
        )
    )
    fine = model_cfg.get("threshold_grid_fine")

    thr_coarse, score_coarse = _threshold_scores(
        y_val,
        y_prob_val,
        coarse,
        policy,
        f_beta,
        min_recall,
    )
    if not fine:
        logger.info(
            "Threshold selected (coarse): policy=%s threshold=%.4f score=%.6f",
            policy,
            thr_coarse,
            score_coarse,
        )
        return thr_coarse, score_coarse

    fine_list = [float(x) for x in fine if 0.0 <= float(x) <= 1.0]
    if not fine_list:
        return thr_coarse, score_coarse
    thr_fine, score_fine = _threshold_scores(
        y_val,
        y_prob_val,
        fine_list,
        policy,
        f_beta,
        min_recall,
    )
    if score_fine >= score_coarse:
        logger.info(
            "Threshold selected (fine): policy=%s threshold=%.4f score=%.6f",
            policy,
            thr_fine,
            score_fine,
        )
        return thr_fine, score_fine
    logger.info(
        "Threshold selected (coarse kept): policy=%s threshold=%.4f score=%.6f",
        policy,
        thr_coarse,
        score_coarse,
    )
    return thr_coarse, score_coarse


def _confusion(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def calibration_bins(y_true: pd.Series, y_prob: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = []
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        if i < bins - 1:
            mask = (y_prob >= left) & (y_prob < right)
        else:
            mask = (y_prob >= left) & (y_prob <= right)
        if mask.sum() == 0:
            avg_pred = 0.0
            pos_rate = 0.0
        else:
            avg_pred = float(np.mean(y_prob[mask]))
            pos_rate = float(np.mean(np.asarray(y_true)[mask]))
        result.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "avg_pred": avg_pred,
                "pos_rate": pos_rate,
                "count": int(mask.sum()),
            }
        )
    return result


def eval_at_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    bins: int = 10,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": _confusion(y_true, y_pred),
        "calibration_bins": calibration_bins(y_true, y_prob, bins=bins),
    }


def fixed_threshold_sweep(
    y_val: pd.Series,
    p_val: np.ndarray,
    y_test: pd.Series,
    p_test: np.ndarray,
    thresholds: list[float] | None = None,
) -> dict:
    thresholds = thresholds or [0.60, 0.70, 0.75, 0.80]
    out = {}
    for thr in thresholds:
        yv = (p_val >= thr).astype(int)
        yt = (p_test >= thr).astype(int)
        out[f"{thr:.2f}"] = {
            "val": {
                "precision": float(precision_score(y_val, yv, zero_division=0)),
                "recall": float(recall_score(y_val, yv, zero_division=0)),
                "f1": float(f1_score(y_val, yv, zero_division=0)),
            },
            "test": {
                "precision": float(precision_score(y_test, yt, zero_division=0)),
                "recall": float(recall_score(y_test, yt, zero_division=0)),
                "f1": float(f1_score(y_test, yt, zero_division=0)),
            },
        }
    return out


def time_stability_metrics(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    p_test: np.ndarray,
    threshold: float,
    chunks: int = 6,
) -> dict:
    chunks = max(1, int(chunks))
    idx_groups = np.array_split(np.arange(len(test_df)), chunks)
    metrics = []
    for i, idx in enumerate(idx_groups, start=1):
        if len(idx) == 0:
            continue
        y_chunk = y_test.iloc[idx]
        p_chunk = p_test[idx]
        if len(set(y_chunk.unique().tolist())) < 2:
            pr_auc = None
        else:
            pr_auc = float(average_precision_score(y_chunk, p_chunk))
        pred = (p_chunk >= threshold).astype(int)
        metrics.append(
            {
                "chunk_id": i,
                "start_time": str(test_df["time"].iloc[idx[0]]),
                "end_time": str(test_df["time"].iloc[idx[-1]]),
                "rows": int(len(idx)),
                "pr_auc": pr_auc,
                "precision": float(precision_score(y_chunk, pred, zero_division=0)),
                "recall": float(recall_score(y_chunk, pred, zero_division=0)),
                "f1": float(f1_score(y_chunk, pred, zero_division=0)),
            }
        )
    return {"chunks": chunks, "metrics_by_chunk": metrics}


def majority_baseline(y: pd.Series) -> dict:
    majority_class = int(y.mean() >= 0.5)
    pred = np.full(len(y), majority_class)
    # For PR-AUC sanity, use constant probabilities = majority positive rate.
    prob = np.full(len(y), float(y.mean()))
    return {
        "majority_positive_rate": float(y.mean()),
        "majority_accuracy": float((pred == y.to_numpy()).mean()),
        "majority_pr_auc": float(average_precision_score(y, prob)),
    }


def top_logreg_coefs(pipeline: Pipeline, feature_columns: list[str], top_n: int = 10) -> dict:
    model = pipeline.named_steps["model"]
    coefs = model.coef_[0]
    pairs = list(zip(feature_columns, coefs))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    top_pos = [{"feature": f, "coef": float(c)} for f, c in pairs_sorted[:top_n]]
    top_neg = [{"feature": f, "coef": float(c)} for f, c in pairs_sorted[-top_n:]]
    return {"top_positive_features": top_pos, "top_negative_features": top_neg}


def main():
    start_time = perf_counter()

    cfg = load_config()
    model_cfg = cfg.get("model1", {})
    run_cfg = cfg.get("run", {})

    if model_cfg.get("type", "logreg") != "logreg":
        raise ValueError("Phase 5 currently supports only model1.type=logreg")

    target_col = model_cfg.get("target_col", "StrongMove")
    model_path = Path(model_cfg.get("output_path", "artifacts/models/model1.pkl"))
    metrics_path = Path(model_cfg.get("metrics_path", "artifacts/reports/model1_metrics.json"))

    train_df, val_df, test_df = _read_splits(cfg)
    logger.info(
        "Split rows loaded: train=%s val=%s test=%s",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = extract_features(
        train_df, val_df, test_df, target_col
    )
    logger.info("Features prepared: count=%s target=%s", len(feature_cols), target_col)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight=model_cfg.get("class_weight", "balanced"),
                    random_state=int(run_cfg.get("seed", 42)),
                    solver=model_cfg.get("solver", "liblinear"),
                    penalty=model_cfg.get("penalty", "l2"),
                    C=float(model_cfg.get("C", 1.0)),
                    max_iter=int(model_cfg.get("max_iter", 1000)),
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    logger.info("LogisticRegression training completed.")

    p_val = pipeline.predict_proba(X_val)[:, 1]
    p_test = pipeline.predict_proba(X_test)[:, 1]
    threshold, best_score = select_threshold(y_val, p_val, model_cfg)

    bins = int(model_cfg.get("calibration_bins", 10))
    val_metrics = eval_at_threshold(y_val, p_val, threshold, bins=bins)
    test_metrics = eval_at_threshold(y_test, p_test, threshold, bins=bins)
    test_metrics["time_stability"] = time_stability_metrics(
        test_df=test_df,
        y_test=y_test,
        p_test=p_test,
        threshold=threshold,
        chunks=int(model_cfg.get("time_stability_chunks", 6)),
    )

    baseline = majority_baseline(y_test)
    sweep = fixed_threshold_sweep(y_val, p_val, y_test, p_test)
    explainability = top_logreg_coefs(pipeline, feature_cols, top_n=10)

    artifact = {
        "pipeline": pipeline,
        "feature_columns": feature_cols,
        "target_col": target_col,
        "threshold_selected": float(threshold),
        "threshold_policy": model_cfg.get("threshold_policy", "f_beta"),
        "threshold_policy_params": {
            "f_beta": float(model_cfg.get("f_beta", 0.5)),
            "min_recall": float(model_cfg.get("min_recall", 0.60)),
        },
        "config_snapshot": {
            "model_type": model_cfg.get("type", "logreg"),
            "class_weight": model_cfg.get("class_weight", "balanced"),
            "solver": model_cfg.get("solver", "liblinear"),
            "penalty": model_cfg.get("penalty", "l2"),
            "C": float(model_cfg.get("C", 1.0)),
            "max_iter": int(model_cfg.get("max_iter", 1000)),
        },
        "split_summary": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "train_time_utc": datetime.now(timezone.utc).isoformat(),
    }

    metrics = {
        "model_type": "logreg",
        "target_col": target_col,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "threshold_selection": {
            "policy": model_cfg.get("threshold_policy", "f_beta"),
            "f_beta": float(model_cfg.get("f_beta", 0.5)),
            "min_recall": float(model_cfg.get("min_recall", 0.60)),
            "grid_coarse": model_cfg.get("threshold_grid_coarse", []),
            "grid_fine": model_cfg.get("threshold_grid_fine", []),
            "selected_threshold": float(threshold),
            "val_best_score": float(best_score),
        },
        "baseline": baseline,
        "val": val_metrics,
        "test": test_metrics,
        "fixed_threshold_sweep": sweep,
        "model_explainability": explainability,
        "artifacts": {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
        },
        "data_summary": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_positive_rate": float(y_train.mean()),
            "val_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    logger.info("Model artifact saved: %s", model_path)
    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=2)
    logger.info("Metrics report saved: %s", metrics_path)
    logger.info(
        "Validation PR-AUC=%.6f | Test PR-AUC=%.6f | Threshold=%.4f",
        val_metrics["pr_auc"],
        test_metrics["pr_auc"],
        threshold,
    )

    # Acceptance check: model load + predict on test works.
    loaded = joblib.load(model_path)
    p_loaded = loaded["pipeline"].predict_proba(X_test[loaded["feature_columns"]])[:, 1]
    if np.any((p_loaded < 0.0) | (p_loaded > 1.0)):
        raise ValueError("Loaded model produced invalid probabilities outside [0,1]")
    logger.info("Artifact load/predict check passed (probabilities in [0,1]).")
    logger.info(
        "Train summary | model=%s | features=%s | threshold=%.4f | "
        "val_pr_auc=%.4f | test_pr_auc=%.4f | test_f1=%.4f | elapsed=%.2fs",
        model_path,
        len(feature_cols),
        threshold,
        val_metrics["pr_auc"],
        test_metrics["pr_auc"],
        test_metrics["f1"],
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()
