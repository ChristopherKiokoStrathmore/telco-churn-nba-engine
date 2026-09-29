import pytest

from telco_nba.nba import IMPLEMENTED_ORDER, decide, eligibility, load_rules, rules_by_id

THRESHOLDS = {
    "churn_high": 0.50,
    "churn_medium": 0.30,
    "clv_high": 1000.0,
    "min_offer_propensity": 0.40,
    "offer_tie_break": "OnlineSecurity",
}


@pytest.fixture(scope="module")
def rule_lookup():
    return rules_by_id(load_rules())


def _decide(rule_lookup, **overrides):
    payload = {
        "churn_probability": 0.1,
        "clv": 100.0,
        "addon_propensities": {"OnlineSecurity": None, "TechSupport": None},
        "eligible": {"OnlineSecurity": False, "TechSupport": False},
        "thresholds": THRESHOLDS,
        "rules_by_id": rule_lookup,
    }
    payload.update(overrides)
    return decide(**payload)


def test_yaml_rule_order_matches_the_implementation():
    document = load_rules()
    assert tuple(rule["id"] for rule in document["rules"]) == IMPLEMENTED_ORDER
    assert document["policy"]["min_offer_propensity"] == pytest.approx(0.40)
    assert document["policy"]["scoring_model"] == "gradient_boosting"


def test_save_call_wins_when_risk_and_clv_are_both_high(rule_lookup):
    decision = _decide(
        rule_lookup,
        churn_probability=0.50,
        clv=1000.0,
        addon_propensities={"OnlineSecurity": 0.9, "TechSupport": 0.9},
        eligible={"OnlineSecurity": True, "TechSupport": True},
    )
    assert decision["action"] == "save_call"
    assert decision["offer"] is None
    assert decision["rule_id"] == "save_call"


def test_high_risk_low_clv_offers_the_higher_propensity_addon(rule_lookup):
    decision = _decide(
        rule_lookup,
        churn_probability=0.80,
        clv=999.0,
        addon_propensities={"OnlineSecurity": 0.10, "TechSupport": 0.15},
        eligible={"OnlineSecurity": True, "TechSupport": True},
    )
    assert decision["action"] == "offer"
    assert decision["offer"] == "TechSupport"
    assert decision["rule_id"] == "offer_high_risk"


def test_high_risk_without_an_eligible_addon_is_no_action(rule_lookup):
    decision = _decide(rule_lookup, churn_probability=0.90, clv=10.0)
    assert decision["action"] == "no_action"
    assert decision["rule_id"] == "no_action"


def test_medium_risk_requires_the_propensity_cutoff(rule_lookup):
    below = _decide(
        rule_lookup,
        churn_probability=0.30,
        clv=10.0,
        addon_propensities={"OnlineSecurity": 0.39, "TechSupport": None},
        eligible={"OnlineSecurity": True, "TechSupport": False},
    )
    assert below["action"] == "no_action"
    above = _decide(
        rule_lookup,
        churn_probability=0.30,
        clv=10.0,
        addon_propensities={"OnlineSecurity": 0.40, "TechSupport": 0.90},
        eligible={"OnlineSecurity": True, "TechSupport": True},
    )
    assert above["offer"] == "TechSupport"
    assert above["rule_id"] == "offer_medium_risk"


def test_propensity_ties_prefer_online_security(rule_lookup):
    decision = _decide(
        rule_lookup,
        churn_probability=0.35,
        addon_propensities={"OnlineSecurity": 0.55, "TechSupport": 0.55},
        eligible={"OnlineSecurity": True, "TechSupport": True},
    )
    assert decision["offer"] == "OnlineSecurity"


def test_below_medium_risk_is_no_action_even_with_a_strong_propensity(rule_lookup):
    decision = _decide(
        rule_lookup,
        churn_probability=0.299,
        addon_propensities={"OnlineSecurity": 0.99, "TechSupport": None},
        eligible={"OnlineSecurity": True, "TechSupport": False},
    )
    assert decision["action"] == "no_action"


def test_eligibility_follows_internet_service_and_current_holding():
    no_internet = eligibility(
        {"InternetService": "No", "OnlineSecurity": "No internet service", "TechSupport": "No internet service"}
    )
    assert no_internet["OnlineSecurity"]["eligible"] is False
    already = eligibility({"InternetService": "DSL", "OnlineSecurity": "Yes", "TechSupport": "No"})
    assert already["OnlineSecurity"]["eligible"] is False
    assert already["TechSupport"]["eligible"] is True
    open_offer = eligibility({"InternetService": "Fiber optic", "OnlineSecurity": "No", "TechSupport": "No"})
    assert open_offer["OnlineSecurity"]["eligible"] is True
    assert open_offer["TechSupport"]["eligible"] is True
