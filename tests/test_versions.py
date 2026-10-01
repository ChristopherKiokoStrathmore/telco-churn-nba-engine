"""Committed JSON records Python major.minor, not the patch version."""

import json
import platform

from telco_nba.model_io import load_bundle
from telco_nba.paths import METRICS_PATH, SCHEMA_PATH
from telco_nba.train import _versions


def test_recorded_python_version_is_major_minor_only():
    recorded = _versions()["python"]
    full = platform.python_version()
    major_minor = ".".join(full.split(".")[:2])
    assert recorded == major_minor
    if full != major_minor:
        assert recorded != full
    metrics = json.loads(METRICS_PATH.read_text())
    schema = json.loads(SCHEMA_PATH.read_text())
    assert metrics["versions"]["python"] == recorded
    assert schema["trained_with"]["python"] == recorded
    bundle = load_bundle()
    assert bundle["feature_schema"]["trained_with"]["python"] == recorded
