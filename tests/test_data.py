import subprocess
import sys

from telco_nba.data import EXPECTED_COLUMNS, file_sha256, load_telco
from telco_nba.paths import CSV_PATH, REPO_ROOT, SHA256_PATH


def test_vendored_csv_matches_checksum_and_row_count():
    digest, name = SHA256_PATH.read_text().strip().split()
    assert name == CSV_PATH.name
    assert file_sha256() == digest
    frame = load_telco()
    assert len(frame) == 7043
    assert list(frame.columns) == EXPECTED_COLUMNS
    blank_total = frame["TotalCharges"].isna()
    assert int(blank_total.sum()) == 11
    assert int((blank_total & (frame["tenure"] == 0)).sum()) == 11
    assert set(frame["Churn"].unique()) == {"Yes", "No"}


def test_download_script_accepts_the_vendored_file():
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "download_data.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "already matches" in completed.stdout
