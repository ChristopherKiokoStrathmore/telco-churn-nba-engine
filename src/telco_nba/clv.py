"""CLV proxy: MonthlyCharges times restricted mean remaining tenure.

The Kaplan-Meier curve is fit on the training split only. Remaining tenure at
time t is the integral of S(u) from t to the last observed training time,
divided by S(t). Nothing is extrapolated past that horizon.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from lifelines import KaplanMeierFitter


def fit_retention_curve(tenure, churned) -> dict[str, list[float]]:
    """Kaplan-Meier survival of tenure, with churn as the event."""
    durations = np.asarray(tenure, dtype=float)
    events = np.asarray(churned, dtype=int)
    fitter = KaplanMeierFitter()
    fitter.fit(durations, event_observed=events)
    survival_frame = fitter.survival_function_
    times = survival_frame.index.to_numpy(dtype=float)
    survival = survival_frame.iloc[:, 0].to_numpy(dtype=float)
    if len(times) == 0:
        raise RuntimeError("Kaplan-Meier curve is empty")
    if times[0] > 0:
        times = np.insert(times, 0, 0.0)
        survival = np.insert(survival, 0, 1.0)
    return {"times": [float(value) for value in times], "survival": [float(value) for value in survival]}


def expected_remaining_months(curve: dict, tenure) -> np.ndarray:
    """Restricted mean remaining lifetime at each tenure. Past the horizon, 0."""
    times = np.asarray(curve["times"], dtype=float)
    survival = np.asarray(curve["survival"], dtype=float)
    tenures = np.asarray(tenure, dtype=float)
    if times.ndim != 1 or survival.shape != times.shape:
        raise ValueError("curve times and survival must be the same length")
    if np.any(np.diff(times) < 0):
        raise ValueError("curve times must be non-decreasing")
    horizon = float(times[-1])
    remaining = np.zeros(len(tenures), dtype=float)
    for index, current in enumerate(tenures):
        if current >= horizon:
            continue
        left = int(np.searchsorted(times, current, side="right") - 1)
        left = max(left, 0)
        survival_now = float(survival[left])
        if survival_now <= 1e-12:
            continue
        area = float(survival[left]) * (float(times[left + 1]) - float(current))
        for step in range(left + 1, len(times) - 1):
            area += float(survival[step]) * (float(times[step + 1]) - float(times[step]))
        remaining[index] = area / survival_now
    return remaining


def clv_proxy(monthly_charges, tenure, curve: dict) -> np.ndarray:
    monthly = np.asarray(monthly_charges, dtype=float)
    remaining = expected_remaining_months(curve, tenure)
    return monthly * remaining
