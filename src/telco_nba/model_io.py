"""Load and save the churn Pipeline for this repo and for responsible-ai-pack.

Project 3 should depend on this contract:

- Model file: artifacts/churn_model.joblib
  A sklearn Pipeline. Preprocessing is inside it. Classes are [0, 1].
  predict_proba(frame)[:, 1] is P(Churn='Yes').
- Schema file: artifacts/feature_schema.json
  feature_order is the exact column list and order.
- Loader: telco_nba.model_io.load_churn_model

The scoring bundle (uptake models, the retention curve, and frozen NBA
thresholds) is artifacts/scoring_bundle.joblib via load_bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from telco_nba.paths import BUNDLE_PATH, MODEL_PATH, SCHEMA_PATH

BUNDLE_KEYS = (
    "churn_models",
    "uptake_models",
    "retention_curve",
    "nba",
    "feature_schema",
    "split",
)


def save_churn_model(pipeline: Pipeline, path: Path | None = None) -> Path:
    destination = path or MODEL_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination, compress=3)
    return destination


def save_feature_schema(schema: dict, path: Path | None = None) -> Path:
    destination = path or SCHEMA_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(schema, indent=2) + "\n")
    return destination


def save_bundle(bundle: dict, path: Path | None = None) -> Path:
    missing = [key for key in BUNDLE_KEYS if key not in bundle]
    if missing:
        raise ValueError(f"Bundle is missing {missing}")
    destination = path or BUNDLE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)
    return destination


def load_bundle(path: Path | None = None) -> dict:
    bundle = joblib.load(path or BUNDLE_PATH)
    missing = [key for key in BUNDLE_KEYS if key not in bundle]
    if missing:
        raise ValueError(f"Bundle is missing {missing}")
    return bundle


def load_churn_model(
    model_path: Path | None = None,
    schema_path: Path | None = None,
) -> tuple[Pipeline, dict]:
    """Return the churn Pipeline and the feature schema.

    Pass a dataframe whose columns are schema['feature_order'], in that order.
    """
    pipeline = joblib.load(model_path or MODEL_PATH)
    schema = json.loads((schema_path or SCHEMA_PATH).read_text())
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected an sklearn Pipeline, found {type(pipeline)}")
    seen = list(pipeline.named_steps["preprocessor"].feature_names_in_)
    expected = list(schema["feature_order"])
    if seen != expected:
        raise ValueError("Saved pipeline columns do not match feature_schema.json")
    if list(pipeline.classes_) != [0, 1]:
        raise ValueError(f"Expected classes [0, 1], found {list(pipeline.classes_)}")
    if schema.get("positive_label") != "Yes":
        raise ValueError("feature schema positive_label must be 'Yes'")
    return pipeline, schema
