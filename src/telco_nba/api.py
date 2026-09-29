"""HTTP scoring API. POST /score accepts one customer's features."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import Response

from telco_nba.schema import CustomerFeatures
from telco_nba.scoring import score_response_json

app = FastAPI(
    title="Telco churn and next-best-action",
    version="0.1.0",
    description=(
        "Churn probability, per-prediction gradient-boosting contributions, "
        "a CLV proxy, and a rule-based next-best action for one customer."
    ),
)


@app.post("/score")
def score(customer: CustomerFeatures) -> Response:
    body = score_response_json(customer.model_dump())
    return Response(content=body, media_type="application/json")
