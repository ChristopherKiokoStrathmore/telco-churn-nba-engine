import json

import numpy as np
import pytest

from telco_nba.data import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    UPTAKE_TARGETS,
    binary_label,
    load_telco,
    uptake_excluded,
)
from telco_nba.metrics import classification_metrics
from telco_nba.model_io import load_bundle, load_churn_model
from telco_nba.paths import METRICS_PATH, SCHEMA_PATH
from telco_nba.pipeline import (
    SCORING_MODEL,
    build_model_pipeline,
    feature_frame,
    positive_proba,
    split_customers,
)
from telco_nba.scoring import nba_summary


def test_new_pipeline_is_unfitted_and_preprocessing_is_inside_it():
    pipeline = build_model_pipeline("logistic_regression")
    numeric_pipe = pipeline.named_steps["preprocessor"].transformers[0][1]
    imputer = numeric_pipe.named_steps["imputer"]
    assert not hasattr(imputer, "statistics_")
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_split_is_disjoint_stratified_and_repeatable():
    frame = load_telco()
    train, test = split_customers(frame)
    again_train, again_test = split_customers(frame)
    assert set(train["customerID"]).isdisjoint(test["customerID"])
    assert len(train) + len(test) == len(frame)
    assert list(train["customerID"]) == list(again_train["customerID"])
    assert list(test["customerID"]) == list(again_test["customerID"])
    train_rate = (train[TARGET_COLUMN] == "Yes").mean()
    test_rate = (test[TARGET_COLUMN] == "Yes").mean()
    assert train_rate == pytest.approx(test_rate, abs=0.02)
    assert TARGET_COLUMN not in feature_frame(train).columns
    assert "customerID" not in feature_frame(train).columns


def test_imputer_and_scaler_statistics_come_from_the_training_rows():
    frame = load_telco()
    train, _test = split_customers(frame)
    pipeline = build_model_pipeline("logistic_regression")
    pipeline.fit(feature_frame(train), binary_label(train[TARGET_COLUMN]))
    numeric = pipeline.named_steps["preprocessor"].named_transformers_["num"]
    imputer = numeric.named_steps["imputer"]
    scaler = numeric.named_steps["scaler"]
    total_index = NUMERIC_FEATURES.index("TotalCharges")
    tenure_index = NUMERIC_FEATURES.index("tenure")
    assert imputer.statistics_[total_index] == pytest.approx(train["TotalCharges"].median())
    assert scaler.mean_[tenure_index] == pytest.approx(train["tenure"].mean())
    assert scaler.mean_[tenure_index] != pytest.approx(frame["tenure"].mean(), abs=1e-6)


def test_saved_models_reproduce_the_metrics_file():
    frame = load_telco()
    train, test = split_customers(frame)
    bundle = load_bundle()
    metrics = json.loads(METRICS_PATH.read_text())
    raw = METRICS_PATH.read_text()
    assert metrics["split"]["n_train"] == len(train)
    assert metrics["split"]["n_test"] == len(test)
    assert "floor(n_test / 10)" in metrics["top_decile_definition"]

    labels = binary_label(test[TARGET_COLUMN])
    features = feature_frame(test)
    for name, pipeline in bundle["churn_models"].items():
        got = classification_metrics(labels, positive_proba(pipeline, features))
        saved = metrics["churn"]["models"][name]
        for key in ("roc_auc", "pr_auc", "top_decile_lift"):
            token = f"{got[key]:.6f}"
            assert token == f"{saved[key]:.6f}"
            assert token in raw
        if name == "dummy_prior":
            train_rate = float((train[TARGET_COLUMN] == "Yes").mean())
            assert np.allclose(positive_proba(pipeline, features), train_rate)

    for target in UPTAKE_TARGETS:
        holdout = test.loc[test["InternetService"] != "No"].reset_index(drop=True)
        uptake_labels = binary_label(holdout[target])
        uptake_features = feature_frame(holdout, uptake_excluded(target))
        block = metrics["uptake"][target]
        assert block["n_test"] == len(holdout)
        assert target not in uptake_features.columns
        assert "MonthlyCharges" not in uptake_features.columns
        assert "TotalCharges" not in uptake_features.columns
        for name, pipeline in bundle["uptake_models"][target].items():
            got = classification_metrics(uptake_labels, positive_proba(pipeline, uptake_features))
            saved = block["models"][name]
            for key in ("roc_auc", "pr_auc", "top_decile_lift"):
                token = f"{got[key]:.6f}"
                assert token == f"{saved[key]:.6f}"
                assert token in raw
            seen = list(pipeline.named_steps["preprocessor"].feature_names_in_)
            assert target not in seen
            assert "MonthlyCharges" not in seen
            assert "TotalCharges" not in seen
            assert TARGET_COLUMN not in seen

    churn_model, schema = load_churn_model()
    assert schema["feature_order"] == FEATURE_COLUMNS
    assert json.loads(SCHEMA_PATH.read_text())["feature_order"] == schema["feature_order"]
    sample = features.head(8)
    from_file = positive_proba(churn_model, sample)
    from_bundle = positive_proba(bundle["churn_models"][SCORING_MODEL], sample)
    assert np.allclose(from_file, from_bundle)

    assert nba_summary(test, bundle, active_only=True) == metrics["nba"]["test_active_customers"]
    assert nba_summary(test, bundle, active_only=False) == metrics["nba"]["test_all_customers"]
