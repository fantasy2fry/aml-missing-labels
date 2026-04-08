from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(func, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return metric value or NaN if sklearn raises due to class edge cases."""
    try:
        return float(func(y_true, y_score))
    except Exception:
        return float("nan")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_pos: np.ndarray,
) -> dict[str, float]:
    """Compute a consistent metric bundle for binary classification."""
    return {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_proba_pos),
        "pr_auc": _safe_metric(average_precision_score, y_true, y_proba_pos),
    }

