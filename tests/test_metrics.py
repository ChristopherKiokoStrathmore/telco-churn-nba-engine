import numpy as np
import pytest

from telco_nba.metrics import classification_metrics, top_decile_stats


def test_constant_scores_have_lift_one_and_auc_half():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
    scores = np.full(len(y), 0.2)
    stats = top_decile_stats(y, scores)
    assert stats["k"] == 1
    assert stats["lift"] == pytest.approx(1.0)
    metrics = classification_metrics(y, scores)
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["pr_auc"] == pytest.approx(y.mean())


def test_perfect_ranking_lifts_above_one():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    stats = top_decile_stats(y, scores)
    assert stats["k"] == 1
    assert stats["top_decile_positive_rate"] == pytest.approx(1.0)
    assert stats["lift"] == pytest.approx(1.0 / 0.2)


def test_reversed_ranking_lifts_below_one():
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    scores = np.linspace(0.0, 1.0, num=10)
    stats = top_decile_stats(y, scores)
    assert stats["lift"] < 1.0


def test_boundary_ties_are_shared_across_the_tie_group():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    scores = np.array([5, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    stats = top_decile_stats(y, scores)
    assert stats["k"] == 1
    # The top score is unique, so the cut does not enter the tie group.
    assert stats["top_decile_positive_rate"] == pytest.approx(1.0)

    tied = np.ones(10)
    shared = top_decile_stats(y, tied)
    assert shared["lift"] == pytest.approx(1.0)
    assert shared["top_decile_positive_rate"] == pytest.approx(y.mean())
