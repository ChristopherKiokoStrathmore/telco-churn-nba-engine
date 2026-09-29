"""Held-out ranking metrics. The top-decile cut is defined here and nowhere else."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TOP_DECILE_DEFINITION = (
    "k = floor(n_test / 10); rows whose score ties cross that cut share it "
    "in proportion to the tie group, so a constant score has lift 1"
)


def top_decile_stats(y_true, y_score) -> dict[str, float | int]:
    """Churn rate in the highest-scored 10 percent, and lift versus the base rate.

    k is floor(n / 10). Sorting is stable. If a group of equal scores crosses
    the cut, each tied row is weighted by need / group size. Constant scores
    therefore have lift 1 rather than an arbitrary slice of the file order.
    """
    y = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    if y.shape != scores.shape or y.ndim != 1:
        raise ValueError("y_true and y_score must be 1-d and the same length")
    if len(y) < 10:
        raise ValueError("top decile needs at least 10 rows")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("y_true must be 0/1")

    k = len(y) // 10
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    scores_sorted = scores[order]
    taken = 0
    weighted_positives = 0.0
    index = 0
    while index < len(y) and taken < k:
        end = index + 1
        while end < len(y) and scores_sorted[end] == scores_sorted[index]:
            end += 1
        group = y_sorted[index:end]
        need = k - taken
        if len(group) <= need:
            weighted_positives += float(group.sum())
            taken += len(group)
        else:
            weighted_positives += float(group.mean()) * need
            taken += need
        index = end

    top_rate = weighted_positives / k
    base_rate = float(y.mean())
    if base_rate <= 0:
        lift = math.nan
    else:
        lift = top_rate / base_rate
    return {
        "k": int(k),
        "top_decile_positive_rate": float(top_rate),
        "base_rate": base_rate,
        "lift": float(lift),
    }


def classification_metrics(y_true, y_score) -> dict[str, float | int]:
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    decile = top_decile_stats(y, scores)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "top_decile_lift": float(decile["lift"]),
        "top_decile_positive_rate": float(decile["top_decile_positive_rate"]),
        "top_decile_k": int(decile["k"]),
        "base_rate": float(decile["base_rate"]),
    }
