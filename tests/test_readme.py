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
FIGURE_PATH = REPO_ROOT / "reports" / "figures" / "churn_roc_pr.png"
# The demo server port is part of the run instructions, not a measured result.
ALLOWED_EXTRA = {"8000", "3000"}
# Six-decimal formatting of round values belongs in metric tables, not prose.
PADDED_PROSE = (
    "72.000000",
    "0.750000",
    "0.500000",
    "0.250000",
    "0.100000",
    "1.000000",
    "0.400000",
)
_MODEL_LABELS = {
    "dummy_prior": "Dummy prior",
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
}


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


def _metric_table_rows(readme: str) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    lines = readme.splitlines()
    header = ["Model", "ROC-AUC", "PR-AUC", "Top-decile lift"]
    index = 0
    while index < len(lines):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if cells[:4] != header:
            index += 1
            continue
        index += 2
        while index < len(lines) and lines[index].strip().startswith("|"):
            body = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
            rows.append((body[0], body[1], body[2], body[3]))
            index += 1
    return rows


def _triple(block: dict) -> tuple[str, str, str]:
    return (
        f"{block['roc_auc']:.6f}",
        f"{block['pr_auc']:.6f}",
        f"{block['top_decile_lift']:.6f}",
    )


def _prose(readme: str) -> str:
    without_code = re.sub(r"```.*?```", "", readme, flags=re.S)
    kept = [line for line in without_code.splitlines() if not line.strip().startswith("|")]
    return "\n".join(kept)


def test_readme_contains_the_saved_score_response():
    body = EXAMPLE_RESPONSE_PATH.read_text().strip()
    assert body in README_PATH.read_text()


def test_readme_numbers_come_from_generated_files():
    allowed = _allowed_text()
    readme = re.sub(r"(?<=\d),(?=\d)", "", README_PATH.read_text())
    numbers = re.findall(r"\d+(?:\.\d+)?", readme)
    missing = sorted({number for number in numbers if number not in allowed and number not in ALLOWED_EXTRA})
    assert missing == []


def test_readme_metric_tables_match_metrics_json():
    metrics = json.loads(METRICS_PATH.read_text())
    rows = _metric_table_rows(README_PATH.read_text())
    assert rows, "README has no ROC-AUC metric tables"

    def matches(block: dict) -> list[tuple[str, str, str, str]]:
        expected = _triple(block)
        return [row for row in rows if row[1:] == expected]

    for name, block in metrics["churn"]["models"].items():
        found = matches(block)
        assert len(found) >= 2, (name, _triple(block))
        assert all(row[0].startswith(_MODEL_LABELS[name]) for row in found)
    for target, block in metrics["uptake"].items():
        for name, model_block in block["models"].items():
            found = matches(model_block)
            assert len(found) >= 1, (target, name, _triple(model_block))
            assert all(row[0].startswith(_MODEL_LABELS[name]) for row in found)


def test_readme_cutoff_table_matches_thresholds():
    metrics = json.loads(METRICS_PATH.read_text())
    table_text = "\n".join(
        line for line in README_PATH.read_text().splitlines() if line.strip().startswith("|")
    )
    thresholds = metrics["nba"]["thresholds"]
    for key in ("churn_high", "churn_medium", "clv_high", "min_offer_propensity"):
        token = f"{thresholds[key]:.6f}"
        assert token in table_text, key
    counts = metrics["nba"]["test_active_customers"]["action_counts"]
    readme = README_PATH.read_text()
    for key in ("save_call", "offer", "no_action", "n"):
        assert str(counts[key]) in readme


def test_readme_prose_does_not_force_padded_six_decimals():
    prose = _prose(README_PATH.read_text())
    present = [token for token in PADDED_PROSE if token in prose]
    assert present == []


def test_readme_source_sentence_sits_above_the_metric_table():
    readme = README_PATH.read_text()
    marker = "The numbers below are copied from"
    assert marker in readme
    assert readme.index(marker) < readme.index("| Model | ROC-AUC |")


def test_readme_links_the_interactive_demo():
    readme = README_PATH.read_text()
    assert "https://telco-churn-nba.vercel.app/demo" in readme
    assert "http://localhost:3000/demo" in readme
    assert "Root Directory" in readme
    assert "/briefing" in readme
    assert "does not run FastAPI" in readme


def test_readme_links_the_model_card_and_embeds_the_figure():
    readme = README_PATH.read_text()
    assert "https://github.com/ChristopherKiokoStrathmore/responsible-ai-pack" in readme
    assert "https://github.com/ChristopherKiokoStrathmore/responsible-ai-pack/blob/main/MODEL_CARD.md" in readme
    assert "will live in the responsible-ai-pack repo" not in readme
    assert "reports/figures/churn_roc_pr.png" in readme
    assert FIGURE_PATH.is_file()
    assert FIGURE_PATH.read_bytes().startswith(b"\x89PNG")
    assert "not Kenyan" in readme or "not Kenyan operator" in readme
    for name in (
        "omnichannel-care-analytics",
        "care-automation-roi",
        "digital-care-roadmap",
    ):
        assert f"https://github.com/ChristopherKiokoStrathmore/{name}" in readme
    run = readme.split("### Run", 1)[1].lstrip()
    assert run.startswith("Requires Python 3.12")
