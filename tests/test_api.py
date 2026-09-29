import json

from fastapi.testclient import TestClient

from telco_nba.api import app
from telco_nba.paths import EXAMPLE_REQUEST_PATH, EXAMPLE_RESPONSE_PATH

client = TestClient(app)


def test_score_example_matches_the_saved_response():
    request = json.loads(EXAMPLE_REQUEST_PATH.read_text())
    response = client.post("/score", json=request)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.text == EXAMPLE_RESPONSE_PATH.read_text()
    body = response.json()
    assert body["scoring_model"] == "gradient_boosting"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["next_best_action"]["action"] in {"save_call", "offer", "no_action"}
    assert len(body["top_reasons"]) == 3
    assert "not SHAP" in body["explanation_method"]
    for reason in body["top_reasons"]:
        if reason["contribution"] > 0:
            assert reason["direction"] == "increases_churn_risk"
        else:
            assert reason["direction"] == "decreases_churn_risk"


def test_missing_total_charges_is_accepted():
    request = json.loads(EXAMPLE_REQUEST_PATH.read_text())
    request["TotalCharges"] = None
    response = client.post("/score", json=request)
    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0


def test_customer_without_internet_is_not_offered_an_addon_score():
    request = json.loads(EXAMPLE_REQUEST_PATH.read_text())
    request["InternetService"] = "No"
    request["OnlineSecurity"] = "No internet service"
    request["OnlineBackup"] = "No internet service"
    request["DeviceProtection"] = "No internet service"
    request["TechSupport"] = "No internet service"
    request["StreamingTV"] = "No internet service"
    request["StreamingMovies"] = "No internet service"
    response = client.post("/score", json=request)
    assert response.status_code == 200
    body = response.json()
    assert body["addon_propensities"]["OnlineSecurity"]["eligible"] is False
    assert body["addon_propensities"]["OnlineSecurity"]["probability"] is None
    assert body["addon_propensities"]["TechSupport"]["probability"] is None


def test_unknown_category_is_rejected():
    request = json.loads(EXAMPLE_REQUEST_PATH.read_text())
    request["gender"] = "Other"
    response = client.post("/score", json=request)
    assert response.status_code == 422
