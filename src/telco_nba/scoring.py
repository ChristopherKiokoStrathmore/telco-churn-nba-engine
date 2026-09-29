"""Score one customer and summarise next-best-action on a frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from telco_nba.clv import clv_proxy, expected_remaining_months
from telco_nba.data import UPTAKE_TARGETS, model_features, uptake_excluded
from telco_nba.explain import explain_log_odds, top_reasons
from telco_nba.model_io import load_bundle
from telco_nba.nba import ADDON_NAMES, decide, eligibility
from telco_nba.pipeline import SCORING_MODEL, feature_frame, positive_proba
from telco_nba.schema import ScoreResponse
from telco_nba.serialize import dumps_rounded

def _r6(value: float) -> float:
    return float(f"{float(value):.6f}")


EXPLANATION_METHOD = (
    "Path contributions along the gradient-boosting trees. "
    "They sum with the model intercept to the churn log-odds for this customer. "
    "They are not SHAP values."
)

_BUNDLE: dict | None = None


def get_bundle() -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = load_bundle()
    return _BUNDLE


def payload_frame(features: dict, columns: list[str]) -> pd.DataFrame:
    row = {}
    for column in columns:
        value = features[column]
        row[column] = np.nan if value is None else value
    frame = pd.DataFrame([row], columns=columns)
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    if "SeniorCitizen" in frame.columns:
        frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(int)
    if "tenure" in frame.columns:
        frame["tenure"] = frame["tenure"].astype(int)
    if "MonthlyCharges" in frame.columns:
        frame["MonthlyCharges"] = frame["MonthlyCharges"].astype(float)
    return frame


def score_customer(features: dict, bundle: dict | None = None) -> dict:
    bundle = bundle or get_bundle()
    schema = bundle["feature_schema"]
    frame = payload_frame(features, list(schema["feature_order"]))
    churn_model = bundle["churn_models"][SCORING_MODEL]
    churn_probability = float(positive_proba(churn_model, frame)[0])
    explanation = explain_log_odds(churn_model, frame)[0]
    reasons = top_reasons(explanation, limit=int(schema.get("top_reasons", 3)))
    for reason in reasons:
        reason["contribution"] = _r6(reason["contribution"])

    flags = eligibility(features)
    addon_scores = {}
    propensities: dict[str, float | None] = {}
    for name in ADDON_NAMES:
        if not flags[name]["eligible"]:
            addon_scores[name] = {
                "eligible": False,
                "probability": None,
                "reason": flags[name]["reason"],
            }
            propensities[name] = None
            continue
        columns = list(bundle["feature_schema"]["uptake_feature_order"][name])
        uptake_frame = payload_frame(features, columns)
        model = bundle["uptake_models"][name][SCORING_MODEL]
        probability = float(positive_proba(model, uptake_frame)[0])
        addon_scores[name] = {"eligible": True, "probability": _r6(probability), "reason": None}
        propensities[name] = probability

    monthly = float(features["MonthlyCharges"])
    remaining = float(expected_remaining_months(bundle["retention_curve"], [features["tenure"]])[0])
    horizon = float(bundle["retention_curve"]["times"][-1])
    monthly_r = _r6(monthly)
    remaining_r = _r6(remaining)
    decision = decide(
        churn_probability=churn_probability,
        clv=monthly * remaining,
        addon_propensities=propensities,
        eligible={name: flags[name]["eligible"] for name in ADDON_NAMES},
        thresholds=bundle["nba"]["thresholds"],
        rules_by_id=bundle["nba"]["rules_by_id"],
    )
    return {
        "churn_probability": _r6(churn_probability),
        "scoring_model": SCORING_MODEL,
        "top_reasons": reasons,
        "next_best_action": decision,
        "addon_propensities": addon_scores,
        "clv_proxy": {
            "monthly_charges": monthly_r,
            "expected_remaining_months": remaining_r,
            "value": _r6(monthly_r * remaining_r),
            "horizon_months": _r6(horizon),
        },
        "explanation_method": EXPLANATION_METHOD,
    }


def score_response_json(features: dict, bundle: dict | None = None) -> str:
    payload = score_customer(features, bundle=bundle)
    rendered = ScoreResponse.model_validate(payload).model_dump()
    return dumps_rounded(rendered)


def nba_summary(frame: pd.DataFrame, bundle: dict, active_only: bool) -> dict:
    """Count actions. active_only keeps rows whose historical Churn label is No.

    The label is not a model feature. It only defines the contact population.
    """
    rows = frame.loc[frame["Churn"] == "No"].reset_index(drop=True) if active_only else frame
    rows = rows.reset_index(drop=True)
    churn_model = bundle["churn_models"][SCORING_MODEL]
    probabilities = positive_proba(churn_model, feature_frame(rows))
    values = clv_proxy(rows["MonthlyCharges"], rows["tenure"], bundle["retention_curve"])
    uptake_scores = {}
    for name in UPTAKE_TARGETS:
        scores = np.full(len(rows), np.nan)
        mask = (rows["InternetService"] != "No") & (rows[name] == "No")
        if bool(mask.any()):
            columns = model_features(uptake_excluded(name))
            scores[mask.to_numpy()] = positive_proba(
                bundle["uptake_models"][name][SCORING_MODEL],
                rows.loc[mask, columns],
            )
        uptake_scores[name] = scores

    counts = {"save_call": 0, "offer": 0, "no_action": 0, "n": int(len(rows))}
    offers = {"OnlineSecurity": 0, "TechSupport": 0}
    thresholds = bundle["nba"]["thresholds"]
    rules = bundle["nba"]["rules_by_id"]
    for index in range(len(rows)):
        customer = rows.iloc[index].to_dict()
        flags = eligibility(customer)
        propensities = {}
        for name in ADDON_NAMES:
            if flags[name]["eligible"]:
                propensities[name] = float(uptake_scores[name][index])
            else:
                propensities[name] = None
        decision = decide(
            churn_probability=float(probabilities[index]),
            clv=float(values[index]),
            addon_propensities=propensities,
            eligible={name: flags[name]["eligible"] for name in ADDON_NAMES},
            thresholds=thresholds,
            rules_by_id=rules,
        )
        counts[decision["action"]] += 1
        if decision["offer"] is not None:
            offers[decision["offer"]] += 1
    return {"action_counts": counts, "offer_counts": offers}
