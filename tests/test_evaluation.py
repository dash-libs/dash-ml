"""Unit tests for threshold checks and model-card rendering (no Spark/MLflow required)."""
from dashml.evaluation import build_model_card, check_thresholds


def test_check_thresholds_pass_and_fail():
    result = check_thresholds({"accuracy": 0.9, "f1_score": 0.5}, {"accuracy": 0.8, "f1_score": 0.7})
    assert result == {"accuracy": True, "f1_score": False}


def test_check_thresholds_missing_metric_fails():
    result = check_thresholds({}, {"accuracy": 0.8})
    assert result == {"accuracy": False}


def test_build_model_card_contains_key_sections():
    card = build_model_card(
        name="churn_predictor",
        version="3",
        model_type="classifier",
        catalog="main",
        schema_name="ml",
        tables=["customers", "orders"],
        target_column="churned",
        feature_columns=["tenure", "spend"],
        params={"n_estimators": 200},
        metrics={"accuracy": 0.91},
        run_id="abc123",
    )
    assert "# Model Card — churn_predictor" in card
    assert "**MLflow run:** `abc123`" in card
    assert "`accuracy`: 0.91" in card
    assert "customers, orders" in card


def test_build_model_card_includes_shap_section_when_present():
    card = build_model_card(
        name="m", version="1", model_type="classifier", catalog="c", schema_name="s",
        tables=[], target_column=None, feature_columns=[], params={}, metrics={},
        shap_by_group={"overall": {"tenure": 0.5, "spend": 0.2}},
    )
    assert "## Feature importance (SHAP)" in card
    assert "`tenure`" in card
