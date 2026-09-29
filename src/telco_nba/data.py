"""Load the vendored IBM telco churn table.

Parsing blank TotalCharges to missing values is a row-wise cast. It does not
learn a statistic, so it is safe to run before the train/test split. Median
imputation stays inside the sklearn Pipeline.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from telco_nba.paths import CSV_PATH, SHA256_PATH

CSV_NAME = "Telco-Customer-Churn.csv"
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"
UPTAKE_TARGETS = ("OnlineSecurity", "TechSupport")
# In this sample the monthly bill is the price of the subscribed bundle, and
# TotalCharges tracks that bill over tenure. Either column would reconstruct
# the add-on holding. They are not uptake features.
UPTAKE_BILLING_COLUMNS = ("MonthlyCharges", "TotalCharges")

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [column for column in FEATURE_COLUMNS if column not in NUMERIC_FEATURES]
EXPECTED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]


def expected_sha256() -> str:
    digest, name = SHA256_PATH.read_text().strip().split()
    if name != CSV_NAME:
        raise ValueError(f"SHA256SUMS names {name!r}, not {CSV_NAME!r}")
    return digest


def file_sha256(path=CSV_PATH) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_telco(path=CSV_PATH) -> pd.DataFrame:
    """Return the churn table with TotalCharges coerced to float."""
    actual = file_sha256(path)
    expected = expected_sha256()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")
    frame = frame.copy()
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(int)
    frame["tenure"] = frame["tenure"].astype(int)
    frame["MonthlyCharges"] = frame["MonthlyCharges"].astype(float)
    other_missing = frame.drop(columns=["TotalCharges"]).isna().any().any()
    if other_missing:
        raise ValueError("Unexpected missing values outside TotalCharges")
    return frame


def categorical_features(excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    return [column for column in CATEGORICAL_FEATURES if column not in excluded]


def model_features(excluded: set[str] | None = None) -> list[str]:
    excluded = excluded or set()
    return [column for column in FEATURE_COLUMNS if column not in excluded]


def uptake_excluded(addon: str) -> set[str]:
    """Columns that must not be used to predict current holding of an add-on."""
    if addon not in UPTAKE_TARGETS:
        raise ValueError(f"Unsupported add-on {addon!r}")
    return {addon, *UPTAKE_BILLING_COLUMNS}


def eligible_for_addon(frame: pd.DataFrame, addon: str) -> pd.DataFrame:
    """Internet customers, whose add-on field is Yes or No rather than 'No internet service'."""
    if addon not in UPTAKE_TARGETS:
        raise ValueError(f"Unsupported add-on {addon!r}")
    part = frame.loc[frame["InternetService"] != "No"].reset_index(drop=True)
    unexpected = set(part[addon].unique()) - {"Yes", "No"}
    if unexpected:
        raise ValueError(f"{addon} has unexpected values among internet customers: {sorted(unexpected)}")
    return part


def binary_label(series: pd.Series, positive: str = POSITIVE_LABEL) -> pd.Series:
    unknown = set(series.unique()) - {positive, "No"}
    if unknown:
        raise ValueError(f"Labels must be {positive!r} or 'No', found {sorted(unknown)}")
    return (series == positive).astype(int)
