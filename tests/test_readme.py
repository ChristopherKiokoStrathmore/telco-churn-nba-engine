"""Every number in the README must appear in a file this repo actually wrote."""

import json
import re

from telco_nba.paths import (
    EXAMPLE_REQUEST_PATH,
    EXAMPLE_RESPONSE_PATH,
    METRICS_PATH,
    REPO_ROOT,
    RULES_PATH,
    SCHEMA_PATH,
    SHA256_PATH,
)

README_PATH = REPO_ROOT / "README.md"
# The demo server port is part of the run instructions, not a measured result.
ALLOWED_EXTRA = {"8000"}


def _allowed_text() -> str:
    parts = [
        METRICS_PATH.read_text(),
        EXAMPLE_RESPONSE_PATH.read_text(),
        EXAMPLE_REQUEST_PATH.read_text(),
        RULES_PATH.read_text(),
        SHA256_PATH.read_text(),
        SCHEMA_PATH.read_text(),
        (REPO_ROOT / "data" / "SOURCE.txt").read_text(),
    ]
    return "\n".join(parts)


def test_readme_contains_the_saved_score_response():
    body = EXAMPLE_RESPONSE_PATH.read_text().strip()
    assert body in README_PATH.read_text()


def test_readme_numbers_come_from_generated_files():
    allowed = _allowed_text()
    readme = re.sub(r"(?<=\d),(?=\d)", "", README_PATH.read_text())
    numbers = re.findall(r"\d+(?:\.\d+)?", readme)
    missing = sorted({number for number in numbers if number not in allowed and number not in ALLOWED_EXTRA})
    assert missing == []


def test_readme_quotes_every_headline_metric():
    metrics = json.loads(METRICS_PATH.read_text())
    readme = README_PATH.read_text()
    for name, block in metrics["churn"]["models"].items():
        for key in ("roc_auc", "pr_auc", "top_decile_lift"):
            token = f"{block[key]:.6f}"
            assert token in readme, (name, key, token)
    for target, block in metrics["uptake"].items():
        for name, model_block in block["models"].items():
            for key in ("roc_auc", "pr_auc", "top_decile_lift"):
                token = f"{model_block[key]:.6f}"
                assert token in readme, (target, name, key, token)
    for token in (
        metrics["nba"]["thresholds"]["churn_high"],
        metrics["nba"]["thresholds"]["churn_medium"],
        metrics["nba"]["thresholds"]["clv_high"],
        metrics["nba"]["thresholds"]["min_offer_propensity"],
    ):
        assert f"{token:.6f}" in readme
    counts = metrics["nba"]["test_active_customers"]["action_counts"]
    for key in ("save_call", "offer", "no_action", "n"):
        assert str(counts[key]) in readme
    assert "responsible-ai-pack" in readme
    assert "not Kenyan" in readme or "not Kenyan operator" in readme
