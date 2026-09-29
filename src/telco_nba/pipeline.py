"""Model constructors. Preprocessing is inside the Pipeline, fit on train only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_nba.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    binary_label,
    categorical_features,
    model_features,
)

SEED = 42
TEST_SIZE = 0.25

MODEL_SPECS: dict[str, dict] = {
    "dummy_prior": {
        "family": "dummy",
        "params": {"strategy": "prior", "random_state": SEED},
    },
    "logistic_regression": {
        "family": "logistic",
        "params": {"solver": "lbfgs", "max_iter": 1000, "random_state": SEED},
    },
    "gradient_boosting": {
        "family": "gradient_boosting",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "random_state": SEED,
        },
    },
}
SCORING_MODEL = "gradient_boosting"


def split_customers(frame: pd.DataFrame, seed: int = SEED, test_size: float = TEST_SIZE):
    """Stratified customer split. Call this before any estimator is fit."""
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame[TARGET_COLUMN],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_preprocessor(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> ColumnTransformer:
    numeric = list(NUMERIC_FEATURES if numeric is None else numeric)
    categorical = list(CATEGORICAL_FEATURES if categorical is None else categorical)
    transformers = []
    if numeric:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric))
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", encoder, categorical))
    if not transformers:
        raise ValueError("A model needs at least one feature column")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_classifier(model_name: str):
    spec = MODEL_SPECS[model_name]
    params = dict(spec["params"])
    family = spec["family"]
    if family == "dummy":
        return DummyClassifier(**params)
    if family == "logistic":
        return LogisticRegression(**params)
    if family == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    raise KeyError(model_name)


def build_model_pipeline(model_name: str, excluded: set[str] | None = None) -> Pipeline:
    """A new unfitted Pipeline. Each model gets its own preprocessor."""
    excluded = excluded or set()
    numeric = [column for column in NUMERIC_FEATURES if column not in excluded]
    categorical = categorical_features(excluded)
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric, categorical)),
            ("classifier", build_classifier(model_name)),
        ]
    )


def feature_frame(frame: pd.DataFrame, excluded: set[str] | None = None) -> pd.DataFrame:
    columns = model_features(excluded)
    return frame.loc[:, columns].copy()


def fit_models(train: pd.DataFrame, label_column: str, excluded: set[str] | None = None) -> dict[str, Pipeline]:
    features = feature_frame(train, excluded)
    labels = binary_label(train[label_column])
    fitted: dict[str, Pipeline] = {}
    for name in MODEL_SPECS:
        pipeline = build_model_pipeline(name, excluded)
        pipeline.fit(features, labels)
        if list(pipeline.classes_) != [0, 1]:
            raise RuntimeError(f"{name} classes were {list(pipeline.classes_)}")
        fitted[name] = pipeline
    return fitted


def positive_proba(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    if list(pipeline.classes_) != [0, 1]:
        raise RuntimeError(f"Expected classes [0, 1], found {list(pipeline.classes_)}")
    return pipeline.predict_proba(features)[:, 1]
