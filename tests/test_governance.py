"""Unit tests for governance artifact builders (no Spark/MLflow required)."""
import json

from dashml.governance import build_governance_artifacts


def test_returns_expected_files():
    files = build_governance_artifacts(
        name="churn_predictor",
        version="3",
        catalog="main",
        schema_name="ml",
        tables=["customers"],
        feature_columns=["tenure"],
        metrics={"accuracy": 0.9},
    )
    assert set(files) == {
        "signature.json", "features.json", "metrics.json",
        "data_sources.yaml", "fairness_report.md", "approval_record.json",
    }


def test_features_json_includes_pii_and_sensitive_columns():
    files = build_governance_artifacts(
        name="m", version="1", catalog="c", schema_name="s", tables=[], feature_columns=["age"],
        metrics={}, sensitive_columns=["gender"], pii_columns=["email"],
    )
    features = json.loads(files["features.json"])
    assert features["sensitive_columns"] == ["gender"]
    assert features["pii_columns"] == ["email"]


def test_metrics_json_computes_baseline_delta():
    files = build_governance_artifacts(
        name="m", version="1", catalog="c", schema_name="s", tables=[], feature_columns=[],
        metrics={"accuracy": 0.9}, baseline_metrics={"accuracy": 0.8},
    )
    metrics = json.loads(files["metrics.json"])
    assert metrics["baseline_comparison"]["accuracy"]["delta"] == 0.1


def test_approval_record_is_pending_with_checklist():
    files = build_governance_artifacts(
        name="m", version="1", catalog="c", schema_name="s", tables=[], feature_columns=[],
        metrics={}, required_approvers=["alice"], checklist=["bias_review"],
    )
    record = json.loads(files["approval_record.json"])
    assert record["state"] == "pending"
    assert record["required_approvers"] == ["alice"]
    assert record["checklist"] == {"bias_review": False}


def test_fairness_report_flags_large_disparity():
    files = build_governance_artifacts(
        name="m", version="1", catalog="c", schema_name="s", tables=[], feature_columns=[],
        metrics={}, sensitive_columns=["age_group"],
        fairness_metrics={"under_30": {"disparity": 0.2}},
    )
    assert "⚠️ review" in files["fairness_report.md"]
