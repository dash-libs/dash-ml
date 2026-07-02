"""Unit tests for drift detection (no Spark required)."""
import numpy as np

from dashml.drift import check_feature_drift, compute_psi, compute_prediction_shift, psi_status


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    values = rng.normal(size=500)
    assert compute_psi(values, values) < 1e-6


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    baseline = rng.normal(loc=0, size=500)
    shifted = rng.normal(loc=5, size=500)
    assert compute_psi(baseline, shifted) > 0.25


def test_psi_categorical_stable():
    baseline = ["a"] * 50 + ["b"] * 50
    same = ["a"] * 50 + ["b"] * 50
    assert compute_psi(baseline, same) < 1e-6


def test_psi_categorical_drifted():
    baseline = ["a"] * 90 + ["b"] * 10
    shifted = ["a"] * 10 + ["b"] * 90
    assert compute_psi(baseline, shifted) > 0.25


def test_psi_returns_zero_below_min_sample_size():
    assert compute_psi([1, 2, 3], [1, 2, 3]) == 0.0


def test_psi_status_buckets():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.18) == "moderate"
    assert psi_status(0.30) == "high_drift"


def test_check_feature_drift_flags_drifted_column():
    rng = np.random.default_rng(0)
    baseline = {"amount": rng.normal(size=500), "stable_col": rng.normal(size=500)}
    production = {"amount": rng.normal(loc=10, size=500), "stable_col": rng.normal(size=500)}

    results = check_feature_drift(baseline, production, threshold=0.15)

    assert results["amount"]["drifted"] is True
    assert results["stable_col"]["drifted"] is False


def test_check_feature_drift_skips_missing_columns():
    results = check_feature_drift({"a": [1, 2, 3]}, {"b": [1, 2, 3]})
    assert results == {}


def test_compute_prediction_shift_no_shift():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    chi2, p_value = compute_prediction_shift(values, values)
    assert p_value > 0.05


def test_compute_prediction_shift_detects_shift():
    rng = np.random.default_rng(0)
    baseline = rng.normal(loc=0, size=300)
    shifted = rng.normal(loc=8, size=300)
    chi2, p_value = compute_prediction_shift(baseline, shifted)
    assert p_value < 0.05
