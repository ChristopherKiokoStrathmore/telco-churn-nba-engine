#!/usr/bin/env python3
"""Draw held-out ROC and precision-recall curves from the committed models.

Uses the saved scoring bundle and the same stratified split as training.
Does not fit a new model. Exits if the curves' ROC-AUC or PR-AUC disagree
with reports/metrics.json at six decimals.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from telco_nba.data import TARGET_COLUMN, binary_label, load_telco
from telco_nba.model_io import load_bundle
from telco_nba.paths import METRICS_PATH, REPO_ROOT
from telco_nba.pipeline import feature_frame, positive_proba, split_customers

FIGURE_PATH = REPO_ROOT / "reports" / "figures" / "churn_roc_pr.png"

_STYLE = {
    "dummy_prior": {"label": "Dummy prior", "color": "#7f7f7f", "lw": 1.6, "ls": "--"},
    "logistic_regression": {"label": "Logistic regression", "color": "#0072B2", "lw": 2.0, "ls": "-"},
    "gradient_boosting": {"label": "Gradient boosting", "color": "#D55E00", "lw": 2.4, "ls": "-"},
}
_ORDER = ("dummy_prior", "logistic_regression", "gradient_boosting")


def _scores(bundle: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frame = load_telco()
    _train, test = split_customers(frame)
    labels = binary_label(test[TARGET_COLUMN]).to_numpy()
    features = feature_frame(test)
    scores = {
        name: positive_proba(pipeline, features)
        for name, pipeline in bundle["churn_models"].items()
    }
    return labels, scores


def _check_against_metrics(labels: np.ndarray, scores: dict[str, np.ndarray], metrics: dict) -> None:
    saved = metrics["churn"]["models"]
    for name, y_score in scores.items():
        roc = f"{float(roc_auc_score(labels, y_score)):.6f}"
        pr = f"{float(average_precision_score(labels, y_score)):.6f}"
        if roc != f"{saved[name]['roc_auc']:.6f}" or pr != f"{saved[name]['pr_auc']:.6f}":
            raise SystemExit(
                f"{name} curve metrics ROC-AUC {roc} PR-AUC {pr} "
                f"do not match reports/metrics.json"
            )


def main() -> None:
    bundle = load_bundle()
    metrics = json.loads(METRICS_PATH.read_text())
    labels, scores = _scores(bundle)
    _check_against_metrics(labels, scores, metrics)
    saved = metrics["churn"]["models"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    roc_ax, pr_ax = axes
    for name in _ORDER:
        style = _STYLE[name]
        y_score = scores[name]
        roc_label = f"{style['label']} ({saved[name]['roc_auc']:.6f})"
        pr_label = f"{style['label']} ({saved[name]['pr_auc']:.6f})"
        plot_kwargs = {
            "color": style["color"],
            "lw": style["lw"],
            "ls": style["ls"],
        }
        if name == "dummy_prior":
            # Constant scores are the chance diagonal and a flat precision line.
            # precision_recall_curve also emits a (recall 0, precision 1) endpoint.
            pr_level = float(saved[name]["pr_auc"])
            roc_ax.plot([0, 1], [0, 1], label=roc_label, **plot_kwargs)
            pr_ax.plot([0, 1], [pr_level, pr_level], label=pr_label, **plot_kwargs)
            continue
        fpr, tpr, _thresholds = roc_curve(labels, y_score)
        precision, recall, _thresholds = precision_recall_curve(labels, y_score)
        roc_ax.plot(fpr, tpr, label=roc_label, **plot_kwargs)
        pr_ax.plot(recall, precision, label=pr_label, **plot_kwargs)

    roc_ax.set_title("ROC")
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    pr_ax.set_title("Precision-recall")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=False,
            fontsize=8,
        )
        axis.grid(True, color="#eeeeee", lw=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.suptitle("Held-out churn ranking on the public IBM telco sample", fontsize=12)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
