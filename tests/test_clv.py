import numpy as np
import pytest
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time

from telco_nba.clv import clv_proxy, expected_remaining_months, fit_retention_curve


def test_remaining_life_on_a_two_step_curve():
    curve = {"times": [0.0, 1.0, 2.0], "survival": [1.0, 0.5, 0.5]}
    remaining = expected_remaining_months(curve, [0, 1, 1.5, 2, 3])
    assert remaining[0] == pytest.approx(1.5)
    assert remaining[1] == pytest.approx(1.0)
    assert remaining[2] == pytest.approx(0.5)
    assert remaining[3] == pytest.approx(0.0)
    assert remaining[4] == pytest.approx(0.0)


def test_remaining_life_at_zero_matches_lifelines_restricted_mean():
    rng = np.random.default_rng(0)
    tenure = rng.integers(1, 40, size=400)
    event = rng.integers(0, 2, size=400)
    curve = fit_retention_curve(tenure, event)
    fitter = KaplanMeierFitter()
    fitter.fit(tenure, event_observed=event)
    horizon = curve["times"][-1]
    restricted = float(restricted_mean_survival_time(fitter, t=horizon))
    assert expected_remaining_months(curve, [0.0])[0] == pytest.approx(restricted, rel=1e-6, abs=1e-6)


def test_clv_is_monthly_charges_times_remaining_tenure():
    curve = {"times": [0.0, 10.0], "survival": [1.0, 1.0]}
    values = clv_proxy([20.0, 5.0], [0.0, 0.0], curve)
    assert values[0] == pytest.approx(200.0)
    assert values[1] == pytest.approx(50.0)


def test_later_tenure_does_not_increase_remaining_life_on_the_sample_curve():
    rng = np.random.default_rng(1)
    tenure = rng.integers(0, 72, size=500)
    event = (rng.random(500) < 0.3).astype(int)
    curve = fit_retention_curve(tenure, event)
    early, late = expected_remaining_months(curve, [0.0, 24.0])
    assert early >= late
