#!/usr/bin/env python3
"""Download the IBM Telco Customer Churn CSV and check its SHA-256.

Source URL (unchanged bytes):
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The parent repository is Apache License 2.0 for its code pattern. IBM does not
state a separate license for the CSV in that repository. See data/SOURCE.txt.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "Telco-Customer-Churn.csv"
SHA_PATH = REPO_ROOT / "data" / "SHA256SUMS"
CHUNK = 1024 * 1024


def expected_sha256() -> str:
    digest, name = SHA_PATH.read_text().strip().split()
    if name != CSV_PATH.name:
        raise SystemExit(f"SHA256SUMS filename is {name!r}, expected {CSV_PATH.name!r}")
    return digest


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    expected = expected_sha256()
    if CSV_PATH.exists() and file_sha256(CSV_PATH) == expected:
        print(f"{CSV_PATH} already matches {expected}")
        return
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()
    CSV_PATH.write_bytes(payload)
    actual = file_sha256(CSV_PATH)
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch: got {actual}, expected {expected}")
    print(f"Wrote {CSV_PATH} ({actual})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — CLI entry point
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
