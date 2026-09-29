"""Rule-based next-best action. The readable source of the rules is the YAML file."""

from __future__ import annotations

from pathlib import Path

import yaml

from telco_nba.paths import RULES_PATH

# Branch order in decide() matches this list. Tests compare it to the YAML.
IMPLEMENTED_ORDER = (
    "save_call",
    "offer_high_risk",
    "offer_medium_risk",
    "no_action",
)
ADDON_NAMES = ("OnlineSecurity", "TechSupport")


def load_rules(path: Path | None = None) -> dict:
    document = yaml.safe_load((path or RULES_PATH).read_text())
    rules = document["rules"]
    ids = tuple(rule["id"] for rule in rules)
    if ids != IMPLEMENTED_ORDER:
        raise ValueError(f"YAML rule order {ids} does not match {IMPLEMENTED_ORDER}")
    priorities = [rule["priority"] for rule in rules]
    if priorities != [1, 2, 3, 4]:
        raise ValueError(f"YAML priorities must be 1..4 in order, found {priorities}")
    policy = document["policy"]
    required = {
        "scoring_model",
        "churn_high_quantile",
        "churn_medium_quantile",
        "clv_high_quantile",
        "min_offer_propensity",
        "oof_folds",
        "quantile_method",
        "offer_tie_break",
    }
    missing = required - set(policy)
    if missing:
        raise ValueError(f"NBA policy is missing {sorted(missing)}")
    if policy["offer_tie_break"] not in ADDON_NAMES:
        raise ValueError("offer_tie_break must name an add-on")
    if policy["churn_medium_quantile"] > policy["churn_high_quantile"]:
        raise ValueError("churn_medium_quantile cannot exceed churn_high_quantile")
    return document


def eligibility(customer: dict) -> dict[str, dict]:
    """Cross-sell eligibility from current holdings. Not a model score."""
    internet = customer["InternetService"] != "No"
    result = {}
    for name in ADDON_NAMES:
        holding = customer[name]
        if not internet:
            result[name] = {
                "eligible": False,
                "reason": "No internet service, so the add-on is not available.",
            }
        elif holding == "Yes":
            result[name] = {
                "eligible": False,
                "reason": f"Customer already has {name}.",
            }
        elif holding == "No":
            result[name] = {"eligible": True, "reason": None}
        else:
            result[name] = {
                "eligible": False,
                "reason": f"{name} is {holding!r}, so the customer is not an eligible cross-sell.",
            }
    return result


def _best_offer(offers: list[tuple[str, float]], tie_break: str) -> str | None:
    if not offers:
        return None

    def sort_key(item: tuple[str, float]) -> tuple[float, int]:
        name, propensity = item
        return (propensity, 1 if name == tie_break else 0)

    return max(offers, key=sort_key)[0]


def decide(
    *,
    churn_probability: float,
    clv: float,
    addon_propensities: dict[str, float | None],
    eligible: dict[str, bool],
    thresholds: dict,
    rules_by_id: dict,
) -> dict:
    """Apply config/nba_rules.yaml in priority order.

    thresholds keys: churn_high, churn_medium, clv_high, min_offer_propensity,
    offer_tie_break.
    """
    high_risk = churn_probability >= thresholds["churn_high"]
    medium_risk = churn_probability >= thresholds["churn_medium"]
    high_clv = clv >= thresholds["clv_high"]
    cutoff = thresholds["min_offer_propensity"]
    tie_break = thresholds["offer_tie_break"]
    available = [
        (name, float(addon_propensities[name]))
        for name in ADDON_NAMES
        if eligible.get(name) and addon_propensities.get(name) is not None
    ]

    if high_risk and high_clv:
        rule = rules_by_id["save_call"]
        return _action(rule, offer=None)

    if high_risk:
        offer = _best_offer(available, tie_break)
        if offer is not None:
            return _action(rules_by_id["offer_high_risk"], offer=offer)

    if medium_risk:
        qualified = [(name, propensity) for name, propensity in available if propensity >= cutoff]
        offer = _best_offer(qualified, tie_break)
        if offer is not None:
            return _action(rules_by_id["offer_medium_risk"], offer=offer)

    return _action(rules_by_id["no_action"], offer=None)


def _action(rule: dict, offer: str | None) -> dict:
    return {
        "action": rule["action"],
        "offer": offer,
        "rule_id": rule["id"],
        "rationale": rule["rationale"],
    }


def rules_by_id(document: dict) -> dict:
    return {rule["id"]: rule for rule in document["rules"]}
