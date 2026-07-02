"""Unit tests for explain_features / _predict_function (needs shap; skipped if absent)."""
import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap")

from dashml.evaluation import _predict_function, explain_features  # noqa: E402


class _PredictModel:
    def predict(self, X):
        return np.asarray(X).sum(axis=1)


class _KNeighborsModel:
    n_neighbors = 3
    n_samples_fit_ = 20

    def kneighbors(self, X, n_neighbors):
        return np.ones((len(X), n_neighbors)), None


class _NoMethodsModel:
    pass


def test_predict_function_prefers_predict():
    fn = _predict_function(_PredictModel(), n_rows=10)
    assert fn(np.array([[1, 2], [3, 4]])).tolist() == [3, 7]


def test_predict_function_falls_back_to_kneighbors():
    fn = _predict_function(_KNeighborsModel(), n_rows=10)
    assert fn(np.zeros((2, 2))).tolist() == [1.0, 1.0]


def test_predict_function_returns_none_for_unsupported_model():
    assert _predict_function(_NoMethodsModel(), n_rows=10) is None


def test_explain_features_returns_none_with_too_few_rows():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert explain_features(_PredictModel(), df) is None


def test_explain_features_returns_importance_per_feature():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=20), "b": rng.normal(size=20)})
    result = explain_features(_PredictModel(), df, sample_size=10, background_size=10)
    assert result is not None
    assert set(result.keys()) == {"a", "b"}


def test_explain_features_returns_none_for_unsupported_model():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=20)})
    assert explain_features(_NoMethodsModel(), df) is None


def test_explain_features_encodes_categorical_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=20), "tier": rng.choice(["gold", "silver"], size=20)})
    result = explain_features(_PredictModel(), df, sample_size=10, background_size=10)
    assert result is not None
    assert "tier" in result
