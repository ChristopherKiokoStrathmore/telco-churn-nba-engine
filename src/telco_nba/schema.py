"""Request and response models for POST /score.

Category sets are the values in the IBM CSV. feature_schema.json, written by
training from the fitted encoder, is checked against these literals in tests.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Gender = Literal["Female", "Male"]
YesNo = Literal["Yes", "No"]
PhoneLines = Literal["Yes", "No", "No phone service"]
Internet = Literal["DSL", "Fiber optic", "No"]
Addon = Literal["Yes", "No", "No internet service"]
Contract = Literal["Month-to-month", "One year", "Two year"]
Payment = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: Gender
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: PhoneLines
    InternetService: Internet
    OnlineSecurity: Addon
    OnlineBackup: Addon
    DeviceProtection: Addon
    TechSupport: Addon
    StreamingTV: Addon
    StreamingMovies: Addon
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: Payment
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


class Reason(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increases_churn_risk", "decreases_churn_risk"]


class NextBestAction(BaseModel):
    action: Literal["save_call", "offer", "no_action"]
    offer: Literal["OnlineSecurity", "TechSupport"] | None
    rule_id: str
    rationale: str


class AddonPropensity(BaseModel):
    eligible: bool
    probability: float | None
    reason: str | None


class AddonScores(BaseModel):
    OnlineSecurity: AddonPropensity
    TechSupport: AddonPropensity


class ClvProxy(BaseModel):
    monthly_charges: float
    expected_remaining_months: float
    value: float
    horizon_months: float


class ScoreResponse(BaseModel):
    churn_probability: float
    scoring_model: Literal["gradient_boosting"]
    top_reasons: list[Reason]
    next_best_action: NextBestAction
    addon_propensities: AddonScores
    clv_proxy: ClvProxy
    explanation_method: str
