import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from telco_nba.data import FEATURE_COLUMNS, load_telco
from telco_nba.explain import explain_log_odds, gradient_boosting_contributions, original_feature
from telco_nba.model_io import load_churn_model
from telco_nba.pipeline import feature_frame, split_customers


def test_path_contributions_reconstruct_the_decision_function():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 3))
    labels = (features[:, 0] + 0.25 * features[:, 1] > 0).astype(int)
    model = GradientBoostingClassifier(
        n_estimators=12,
        max_depth=2,
        learning_rate=0.1,
        random_state=0,
    )
    model.fit(features, labels)
    bias, contributions = gradient_boosting_contributions(model, features)
    reconstructed = bias + contributions.sum(axis=1)
    assert np.allclose(reconstructed, model.decision_function(features), atol=1e-8)


def test_saved_churn_model_contributions_sum_to_its_log_odds():
    frame = load_telco()
    _train, test = split_customers(frame)
    pipeline, _schema = load_churn_model()
    sample = feature_frame(test).head(4)
    explanations = explain_log_odds(pipeline, sample)
    decisions = pipeline.decision_function(sample)
    encoded = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    assert {original_feature(name, FEATURE_COLUMNS) for name in encoded} == set(FEATURE_COLUMNS)
    for explanation, decision in zip(explanations, decisions):
        total = explanation.bias + sum(explanation.contributions.values())
        assert total == pytest.approx(decision, abs=1e-6)
        assert set(explanation.contributions) == set(FEATURE_COLUMNS)
