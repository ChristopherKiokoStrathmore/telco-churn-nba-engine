"""Fit the held-out churn, uptake, CLV, and next-best-action artefacts.

The customer split is drawn before any estimator, imputer, scaler, encoder, or
Kaplan-Meier curve is fit. Reported metrics are computed on the held-out
customers from the models saved in artifacts/.
"""

from __future__ import annotations

import platform

import fastapi
import lifelines
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from telco_nba import __version__
from telco_nba.clv import clv_proxy, expected_remaining_months, fit_retention_curve
from telco_nba.data import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    UPTAKE_TARGETS,
    binary_label,
    eligible_for_addon,
    expected_sha256,
    load_telco,
    model_features,
    uptake_excluded,
)
from telco_nba.metrics import TOP_DECILE_DEFINITION, classification_metrics
from telco_nba.model_io import load_bundle, save_bundle, save_churn_model, save_feature_schema
from telco_nba.nba import load_rules, rules_by_id
from telco_nba.paths import (
    EXAMPLE_REQUEST_PATH,
    EXAMPLE_RESPONSE_PATH,
    METRICS_PATH,
    SOURCE_REPOSITORY,
    SOURCE_URL,
)
from telco_nba.pipeline import (
    MODEL_SPECS,
    SCORING_MODEL,
    SEED,
    TEST_SIZE,
    build_model_pipeline,
    feature_frame,
    fit_models,
    positive_proba,
    split_customers,
)
from telco_nba.scoring import nba_summary, score_response_json
from telco_nba.serialize import dumps_rounded


def _versions() -> dict[str, str]:
    return {
        "telco_nba": __version__,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "lifelines": lifelines.__version__,
        "fastapi": fastapi.__version__,
    }


def _profile(frame: pd.DataFrame) -> dict:
    internet = frame["InternetService"] != "No"
    churn_yes = frame[TARGET_COLUMN] == "Yes"
    blank_total = frame["TotalCharges"].isna()
    return {
        "name": "IBM Telco Customer Churn",
        "source_url": SOURCE_URL,
        "repository": SOURCE_REPOSITORY,
        "repository_license": "Apache-2.0",
        "data_license_note": (
            "Apache-2.0 on the IBM repository covers that code pattern. "
            "The repository does not state a separate license for the CSV. "
            "This repo vendors the file unchanged."
        ),
        "sha256": expected_sha256(),
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "blank_total_charges": int(blank_total.sum()),
        "blank_total_charges_at_tenure_0": int((blank_total & (frame["tenure"] == 0)).sum()),
        "churn_yes": int(churn_yes.sum()),
        "churn_rate": float(churn_yes.mean()),
        "internet_customers": int(internet.sum()),
        "online_security_yes_among_internet": int((internet & (frame["OnlineSecurity"] == "Yes")).sum()),
        "tech_support_yes_among_internet": int((internet & (frame["TechSupport"] == "Yes")).sum()),
    }


def _evaluate(models: dict, features: pd.DataFrame, labels: pd.Series) -> dict:
    report = {}
    for name, pipeline in models.items():
        report[name] = classification_metrics(labels, positive_proba(pipeline, features))
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "n_iter_"):
            n_iter = int(np.max(classifier.n_iter_))
            report[name]["n_iter"] = n_iter
            if n_iter >= int(classifier.max_iter):
                raise RuntimeError(f"{name} did not converge (n_iter={n_iter})")
    return report


def _oof_scoring_proba(train: pd.DataFrame, n_splits: int) -> np.ndarray:
    """Out-of-fold churn probabilities on the training split only."""
    pipeline = build_model_pipeline(SCORING_MODEL)
    features = feature_frame(train)
    labels = binary_label(train[TARGET_COLUMN])
    folder = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    predicted = cross_val_predict(
        pipeline,
        features,
        labels,
        cv=folder,
        method="predict_proba",
        n_jobs=1,
    )
    return predicted[:, 1]


def _categories(pipeline) -> dict[str, list]:
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    categories: dict[str, list] = {}
    for name, values in zip(encoder.feature_names_in_, encoder.categories_):
        cleaned = []
        for value in values:
            if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
                cleaned.append(int(value))
            else:
                cleaned.append(str(value))
        categories[str(name)] = cleaned
    return categories


def _feature_schema(churn_model, uptake_models: dict, versions: dict) -> dict:
    seen = list(churn_model.named_steps["preprocessor"].feature_names_in_)
    if seen != FEATURE_COLUMNS:
        raise RuntimeError(f"Churn model columns {seen} != {FEATURE_COLUMNS}")
    uptake_order = {}
    for name, models in uptake_models.items():
        columns = list(models[SCORING_MODEL].named_steps["preprocessor"].feature_names_in_)
        expected = model_features(uptake_excluded(name))
        if columns != expected:
            raise RuntimeError(f"{name} columns {columns} != {expected}")
        if name in columns or TARGET_COLUMN in columns or ID_COLUMN in columns:
            raise RuntimeError(f"{name} model includes a forbidden column")
        uptake_order[name] = columns
    return {
        "model_path": "artifacts/churn_model.joblib",
        "bundle_path": "artifacts/scoring_bundle.joblib",
        "load_function": "telco_nba.model_io.load_churn_model",
        "predict": "predict_proba(frame)[:, 1] is P(Churn='Yes')",
        "positive_label": "Yes",
        "positive_class_index": 1,
        "id_column_excluded": ID_COLUMN,
        "target_excluded": TARGET_COLUMN,
        "feature_order": list(FEATURE_COLUMNS),
        "numeric_features": ["tenure", "MonthlyCharges", "TotalCharges"],
        "categorical_features": [column for column in FEATURE_COLUMNS if column not in ("tenure", "MonthlyCharges", "TotalCharges")],
        "categories": _categories(churn_model),
        "top_reasons": 3,
        "total_charges_missing": (
            "Blank TotalCharges cells are parsed to NaN before the split. "
            "Median imputation is a step inside the sklearn Pipeline and is fit on the training rows only."
        ),
        "uptake_feature_order": uptake_order,
        "uptake_note": (
            "Uptake models predict current holding of OnlineSecurity or TechSupport "
            "among customers with internet service. They are not campaign-response models. "
            "The target column is excluded. MonthlyCharges and TotalCharges are excluded "
            "because in this sample the bill is the price of the subscribed bundle, so "
            "those fields would reconstruct the holding."
        ),
        "split": {"seed": SEED, "test_size": TEST_SIZE, "stratify": TARGET_COLUMN},
        "trained_with": versions,
    }


def _payload(row: pd.Series) -> dict:
    payload = {}
    for column in FEATURE_COLUMNS:
        value = row[column]
        if pd.isna(value):
            payload[column] = None
        elif column in ("SeniorCitizen", "tenure"):
            payload[column] = int(value)
        elif column in ("MonthlyCharges", "TotalCharges"):
            payload[column] = float(value)
        else:
            payload[column] = str(value)
    return payload


def _params_for_metrics() -> dict:
    exported = {}
    for name, spec in MODEL_SPECS.items():
        exported[name] = {"family": spec["family"], "params": dict(spec["params"])}
    return exported


def train() -> dict:
    document = load_rules()
    policy = document["policy"]
    frame = load_telco()
    train_rows, test_rows = split_customers(frame)
    churn_models = fit_models(train_rows, TARGET_COLUMN)
    churn_report = _evaluate(
        churn_models,
        feature_frame(test_rows),
        binary_label(test_rows[TARGET_COLUMN]),
    )

    oof = _oof_scoring_proba(train_rows, int(policy["oof_folds"]))
    method = str(policy["quantile_method"])
    churn_high = float(np.quantile(oof, policy["churn_high_quantile"], method=method))
    churn_medium = float(np.quantile(oof, policy["churn_medium_quantile"], method=method))
    if not churn_medium <= churn_high:
        raise RuntimeError("Resolved churn_medium cutoff exceeds churn_high")

    curve = fit_retention_curve(train_rows["tenure"], binary_label(train_rows[TARGET_COLUMN]))
    train_clv = clv_proxy(train_rows["MonthlyCharges"], train_rows["tenure"], curve)
    clv_high = float(np.quantile(train_clv, policy["clv_high_quantile"], method=method))
    remaining_at_zero = float(expected_remaining_months(curve, [0.0])[0])

    uptake_models = {}
    uptake_report = {}
    for target in UPTAKE_TARGETS:
        uptake_train = eligible_for_addon(train_rows, target)
        uptake_test = eligible_for_addon(test_rows, target)
        excluded = uptake_excluded(target)
        fitted = fit_models(uptake_train, target, excluded=excluded)
        labels = binary_label(uptake_test[target])
        uptake_models[target] = fitted
        uptake_report[target] = {
            "population": "InternetService != No, using the same customer split as the churn model",
            "label": (
                f"current holding of {target} (Yes=1). Not a campaign response. "
                "MonthlyCharges and TotalCharges are excluded."
            ),
            "excluded_features": sorted(excluded),
            "n_train": int(len(uptake_train)),
            "n_test": int(len(uptake_test)),
            "test_base_rate": float(labels.mean()),
            "models": _evaluate(fitted, feature_frame(uptake_test, excluded), labels),
        }

    versions = _versions()
    schema = _feature_schema(churn_models[SCORING_MODEL], uptake_models, versions)
    thresholds = {
        "churn_high": churn_high,
        "churn_medium": churn_medium,
        "clv_high": clv_high,
        "min_offer_propensity": float(policy["min_offer_propensity"]),
        "offer_tie_break": policy["offer_tie_break"],
    }
    bundle = {
        "churn_models": churn_models,
        "uptake_models": uptake_models,
        "retention_curve": curve,
        "nba": {
            "thresholds": thresholds,
            "rules": document["rules"],
            "rules_by_id": rules_by_id(document),
            "policy": policy,
        },
        "feature_schema": schema,
        "split": {"seed": SEED, "test_size": TEST_SIZE, "stratify": TARGET_COLUMN},
    }
    save_churn_model(churn_models[SCORING_MODEL])
    save_feature_schema(schema)
    save_bundle(bundle)
    reloaded = load_bundle()

    active_actions = nba_summary(test_rows, reloaded, active_only=True)
    all_actions = nba_summary(test_rows, reloaded, active_only=False)

    example_row = test_rows.loc[test_rows["TotalCharges"].notna()].iloc[0]
    request = _payload(example_row)
    EXAMPLE_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLE_REQUEST_PATH.write_text(dumps_rounded(request))
    EXAMPLE_RESPONSE_PATH.write_text(score_response_json(request, bundle=reloaded))

    roc_order = sorted(churn_report, key=lambda name: churn_report[name]["roc_auc"], reverse=True)
    metrics = {
        "dataset": _profile(frame),
        "split": {
            "seed": SEED,
            "test_size": TEST_SIZE,
            "stratify": TARGET_COLUMN,
            "n_train": int(len(train_rows)),
            "n_test": int(len(test_rows)),
            "train_churn_rate": float((train_rows[TARGET_COLUMN] == "Yes").mean()),
            "test_churn_rate": float((test_rows[TARGET_COLUMN] == "Yes").mean()),
        },
        "top_decile_definition": TOP_DECILE_DEFINITION,
        "model_params": _params_for_metrics(),
        "scoring_model": SCORING_MODEL,
        "churn": {
            "positive_label": "Yes",
            "test_base_rate": churn_report[SCORING_MODEL]["base_rate"],
            "roc_auc_rank_high_to_low": roc_order,
            "models": churn_report,
        },
        "uptake": uptake_report,
        "clv": {
            "definition": (
                "MonthlyCharges multiplied by restricted mean remaining tenure from a "
                "Kaplan-Meier curve fit on the training split only. The integral stops "
                "at the last observed training tenure and is not extrapolated."
            ),
            "horizon_months": float(curve["times"][-1]),
            "expected_remaining_months_at_tenure_0": remaining_at_zero,
            "train_clv_p50": float(np.quantile(train_clv, 0.50, method=method)),
            "train_clv_p75": float(np.quantile(train_clv, 0.75, method=method)),
        },
        "nba": {
            "threshold_source": (
                "churn_high and churn_medium are quantiles of 3-fold out-of-fold "
                "gradient-boosting probabilities on the training split. clv_high is "
                "the matching quantile of the training-split CLV proxy. "
                "min_offer_propensity is the fixed policy constant in config/nba_rules.yaml. "
                "Cutoffs are not chosen on the test set."
            ),
            "policy": policy,
            "thresholds": thresholds,
            "test_active_customers": active_actions,
            "test_all_customers": all_actions,
            "active_definition": "Held-out customers whose historical Churn label is No. The label is not a model input.",
        },
        "example_customer_id": str(example_row[ID_COLUMN]),
        "example_customer_historical_churn": str(example_row[TARGET_COLUMN]),
        "versions": versions,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(dumps_rounded(metrics))
    print(f"Wrote {METRICS_PATH}")
    for name in roc_order:
        row = churn_report[name]
        print(
            f"churn {name}: ROC-AUC {row['roc_auc']:.6f}  "
            f"PR-AUC {row['pr_auc']:.6f}  lift {row['top_decile_lift']:.6f}"
        )
    print(
        "NBA active counts",
        active_actions["action_counts"],
        "offers",
        active_actions["offer_counts"],
    )
    return metrics


if __name__ == "__main__":
    train()
