"""The browser scoring document must match POST /score on the public sample."""

import json

from telco_nba.data import load_telco
from telco_nba.metrics import top_decile_stats
from telco_nba.pipeline import split_customers
from telco_nba.portable import (
    WEB_EXAMPLES,
    WEB_HOLDOUT,
    WEB_MODEL,
    WEB_PARITY,
    public_score,
    render_web_demo_files,
    score_portable,
)
from telco_nba.paths import REPO_ROOT
from telco_nba.scoring import score_customer


def _document():
    files = render_web_demo_files()
    return json.loads(files[WEB_MODEL]), files


def test_portable_scores_match_the_api_on_the_holdout_and_examples():
    document, files = _document()
    parity = json.loads(files[WEB_PARITY])
    frame = load_telco()
    _train, test = split_customers(frame)
    assert parity["n"] == len(test) == 1761
    assert len(files[WEB_HOLDOUT].splitlines()) == 1762

    labels = []
    probabilities = []
    for row, expected in zip(test.to_dict(orient="records"), parity["rows"], strict=True):
        features = {}
        for column, value in row.items():
            if column in ("customerID", "Churn"):
                continue
            if column == "TotalCharges" and value != value:
                features[column] = None
            else:
                features[column] = value
        api = score_customer(features)
        portable = public_score(document, features)
        assert portable == api
        assert expected["customer_id"] == row["customerID"]
        assert expected["action"] == api["next_best_action"]["action"]
        assert expected["churn_probability"] == api["churn_probability"]
        labels.append(1 if row["Churn"] == "Yes" else 0)
        probabilities.append(score_portable(document, features)["_unrounded_churn_probability"])

    lift = top_decile_stats(labels, probabilities)
    assert f"{lift['lift']:.6f}" == "2.806733"
    assert f"{parity['lift']:.6f}" == "2.806733"

    for example in parity["examples"]:
        assert public_score(document, example["features"]) == example["response"]
        if example["source"] == "examples/score_request.json":
            committed = json.loads((REPO_ROOT / "examples/score_response.json").read_text())
            assert example["response"] == committed


def test_committed_web_demo_files_match_the_export():
    expected = render_web_demo_files()
    for relative, text in expected.items():
        path = REPO_ROOT / relative
        assert path.is_file(), relative
        assert path.read_text() == text, relative
