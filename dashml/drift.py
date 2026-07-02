"""Distribution-shift detection — pure numpy/pandas, no Spark dependency.

Kept separate from metrics.py so the math is unit-testable without a
SparkSession; metrics.py collects Spark columns into arrays and calls in here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

STABLE, MODERATE, HIGH_DRIFT = "stable", "moderate", "high_drift"


def compute_psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index. <0.10 stable, 0.10-0.25 moderate, >0.25 high drift."""
    exp_s, act_s = pd.Series(expected), pd.Series(actual)
    if not pd.api.types.is_numeric_dtype(exp_s) or not pd.api.types.is_numeric_dtype(act_s):
        return _categorical_psi(exp_s, act_s)

    expected_arr = exp_s.dropna().to_numpy(dtype=float)
    actual_arr = act_s.dropna().to_numpy(dtype=float)
    if len(expected_arr) < 10 or len(actual_arr) < 10:
        return 0.0

    breakpoints = np.unique(np.percentile(expected_arr, np.linspace(0, 100, bins + 1)))
    # Extend the outer edges to +/-inf so actual values outside the expected
    # range land in the extreme bins instead of being dropped by np.histogram
    # — a distribution that's moved entirely outside the training range is
    # the worst kind of drift and must not silently compute as "no change".
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    exp_counts = np.histogram(expected_arr, bins=breakpoints)[0]
    act_counts = np.histogram(actual_arr, bins=breakpoints)[0]

    eps = 1e-6
    exp_pct = (exp_counts + eps) / (exp_counts.sum() + eps * len(exp_counts))
    act_pct = (act_counts + eps) / (act_counts.sum() + eps * len(act_counts))
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    exp, act = expected.dropna().astype(str), actual.dropna().astype(str)
    if len(exp) < 10 or len(act) < 10:
        return 0.0
    eps = 1e-6
    exp_pct, act_pct = exp.value_counts(normalize=True), act.value_counts(normalize=True)
    psi = 0.0
    for cat in set(exp_pct.index) | set(act_pct.index):
        e, a = float(exp_pct.get(cat, 0.0)) + eps, float(act_pct.get(cat, 0.0)) + eps
        psi += (a - e) * np.log(a / e)
    return float(psi)


def psi_status(psi: float, threshold: float = 0.15) -> str:
    if psi < min(0.10, threshold):
        return STABLE
    if psi < max(0.25, threshold):
        return MODERATE
    return HIGH_DRIFT


def compute_prediction_shift(expected, actual) -> tuple[float, float]:
    """Chi-squared test on binned prediction distributions. Returns (statistic, p_value)."""
    from scipy.stats import chi2_contingency

    expected_arr, actual_arr = np.asarray(expected), np.asarray(actual)
    bins = np.unique(np.percentile(np.concatenate([expected_arr, actual_arr]), np.linspace(0, 100, 11)))
    exp_hist = np.histogram(expected_arr, bins=bins)[0]
    act_hist = np.histogram(actual_arr, bins=bins)[0]

    mask = (exp_hist + act_hist) > 0
    if mask.sum() < 2:
        return 0.0, 1.0
    try:
        chi2, p_value, _, _ = chi2_contingency(np.array([exp_hist[mask], act_hist[mask]]))
        return float(chi2), float(p_value)
    except Exception:
        return 0.0, 1.0


def check_feature_drift(
    baseline: dict[str, "np.ndarray | list"],
    production: dict[str, "np.ndarray | list"],
    threshold: float = 0.15,
) -> dict[str, dict]:
    """Per-column PSI drift check. baseline/production: {column: values}."""
    results = {}
    for col, base_vals in baseline.items():
        if col not in production:
            continue
        try:
            psi = compute_psi(base_vals, production[col])
            results[col] = {"psi": round(psi, 4), "status": psi_status(psi, threshold), "drifted": psi > threshold}
        except Exception as e:
            results[col] = {"psi": None, "error": str(e)}
    return results
