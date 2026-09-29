"""Per-prediction gradient-boosting contributions.

For each tree, the leaf value equals the root value plus the changes along the
path. Summing those changes, times the learning rate, and adding the initial
log-odds, reconstructs decision_function. One-hot columns are then added back
together under the original feature name.

This is a path decomposition of this model, not a SHAP value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class LogOddsExplanation:
    bias: float
    contributions: dict[str, float]
    decision: float


def gradient_boosting_contributions(
    model: GradientBoostingClassifier, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return bias (n,) and per-feature contributions (n, p) in log-odds."""
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be 2-d")
    if list(model.classes_) != [0, 1]:
        raise RuntimeError(f"Expected classes [0, 1], found {list(model.classes_)}")
    prior = model.init_.predict_proba(matrix)[:, 1]
    prior = np.clip(prior, 1e-12, 1 - 1e-12)
    bias = np.log(prior / (1.0 - prior))
    contributions = np.zeros(matrix.shape, dtype=float)
    learning_rate = float(model.learning_rate)
    for estimator in model.estimators_[:, 0]:
        tree = estimator.tree_
        bias = bias + learning_rate * float(tree.value[0, 0, 0])
        for row_index in range(matrix.shape[0]):
            node = 0
            row = matrix[row_index]
            while tree.children_left[node] != -1:
                feature = int(tree.feature[node])
                threshold = tree.threshold[node]
                if np.isnan(row[feature]):
                    raise ValueError("Cannot explain a missing transformed feature")
                child = tree.children_left[node] if row[feature] <= threshold else tree.children_right[node]
                delta = float(tree.value[child, 0, 0] - tree.value[node, 0, 0])
                contributions[row_index, feature] += learning_rate * delta
                node = child
    return bias, contributions


def original_feature(encoded_name: str, columns: list[str]) -> str:
    bare = encoded_name.split("__", 1)[1] if "__" in encoded_name else encoded_name
    matches = [column for column in columns if bare == column or bare.startswith(column + "_")]
    if not matches:
        raise ValueError(f"Cannot map encoded feature {encoded_name!r}")
    return max(matches, key=len)


def explain_log_odds(pipeline: Pipeline, frame: pd.DataFrame) -> list[LogOddsExplanation]:
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, GradientBoostingClassifier):
        raise TypeError("explain_log_odds is implemented for gradient boosting")
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = np.asarray(preprocessor.transform(frame), dtype=float)
    encoded_names = list(preprocessor.get_feature_names_out())
    if transformed.shape[1] != len(encoded_names):
        raise RuntimeError("Transformed width does not match feature names")
    bias, contributions = gradient_boosting_contributions(classifier, transformed)
    columns = list(frame.columns)
    grouped_index = {column: index for index, column in enumerate(columns)}
    grouped = np.zeros((len(frame), len(columns)), dtype=float)
    for encoded_index, encoded_name in enumerate(encoded_names):
        column = original_feature(encoded_name, columns)
        grouped[:, grouped_index[column]] += contributions[:, encoded_index]
    decisions = np.asarray(pipeline.decision_function(frame), dtype=float).reshape(-1)
    explanations = []
    for row_index in range(len(frame)):
        row_contributions = {
            column: float(grouped[row_index, column_index]) for column_index, column in enumerate(columns)
        }
        explanations.append(
            LogOddsExplanation(
                bias=float(bias[row_index]),
                contributions=row_contributions,
                decision=float(decisions[row_index]),
            )
        )
    return explanations


def top_reasons(explanation: LogOddsExplanation, limit: int = 3) -> list[dict]:
    ranked = sorted(
        explanation.contributions.items(),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    reasons = []
    for feature, contribution in ranked:
        if contribution == 0:
            continue
        reasons.append(
            {
                "feature": feature,
                "contribution": contribution,
                "direction": "increases_churn_risk" if contribution > 0 else "decreases_churn_risk",
            }
        )
        if len(reasons) == limit:
            break
    return reasons
