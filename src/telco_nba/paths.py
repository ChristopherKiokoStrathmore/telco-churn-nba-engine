"""Repository paths used by training, the API, and the tests."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CSV_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
SHA256_PATH = DATA_DIR / "SHA256SUMS"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_model.joblib"
BUNDLE_PATH = ARTIFACT_DIR / "scoring_bundle.joblib"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
METRICS_PATH = REPO_ROOT / "reports" / "metrics.json"
RULES_PATH = REPO_ROOT / "config" / "nba_rules.yaml"
EXAMPLE_REQUEST_PATH = REPO_ROOT / "examples" / "score_request.json"
EXAMPLE_RESPONSE_PATH = REPO_ROOT / "examples" / "score_response.json"

SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
SOURCE_REPOSITORY = "https://github.com/IBM/telco-customer-churn-on-icp4d"
