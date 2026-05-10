"""
Utilities for out-of-fold predictions used by error and threshold analysis.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from train_breast_cancer_models import RANDOM_SEED, build_models


def _positive_scores(estimator, X_test):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
        return 1.0 / (1.0 + np.exp(-scores))
    return estimator.predict(X_test).astype(float)


def get_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: list[str],
    outer_splits: int = 5,
    inner_splits: int = 3,
    scoring: str = "roc_auc",
) -> dict[str, pd.DataFrame]:
    """Return out-of-fold predictions for selected models."""
    all_models = build_models()
    missing = [m for m in model_names if m not in all_models]
    if missing:
        raise ValueError(f"Unknown model names: {missing}")

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_SEED)

    outputs: dict[str, pd.DataFrame] = {}

    for model_name in model_names:
        spec = all_models[model_name]
        rows = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            estimator = clone(spec["estimator"])
            param_grid = spec["param_grid"]

            if param_grid:
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    scoring=scoring,
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True,
                )
                search.fit(X_train, y_train)
                fitted = search.best_estimator_
                best_params = search.best_params_
            else:
                fitted = estimator.fit(X_train, y_train)
                best_params = {}

            y_score = _positive_scores(fitted, X_test)
            y_pred = (y_score >= 0.5).astype(int)

            for local_i, original_idx in enumerate(test_idx):
                true_label = int(y_test.iloc[local_i])
                pred_label = int(y_pred[local_i])
                if true_label == 1 and pred_label == 1:
                    error_type = "TP"
                elif true_label == 0 and pred_label == 0:
                    error_type = "TN"
                elif true_label == 0 and pred_label == 1:
                    error_type = "FP"
                else:
                    error_type = "FN"

                rows.append({
                    "row_index": int(original_idx),
                    "fold": int(fold_idx),
                    "model_name": model_name,
                    "y_true": true_label,
                    "y_score": float(y_score[local_i]),
                    "y_pred": pred_label,
                    "error_type": error_type,
                    "distance_from_threshold": float(abs(y_score[local_i] - 0.5)),
                    "best_params": str(best_params),
                })

        out = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
        outputs[model_name] = out

    return outputs


def summarize_predictions(pred_df: pd.DataFrame, threshold: float = 0.5) -> dict[str, float]:
    y_true = pred_df["y_true"].to_numpy()
    y_score = pred_df["y_score"].to_numpy()
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
