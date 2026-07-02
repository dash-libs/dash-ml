"""Unit tests for format_comparison_table (pure formatting, no MLflow required)."""
from dashml.experiment import format_comparison_table


def test_empty_runs_message():
    assert format_comparison_table([]) == "No runs found."


def test_table_includes_header_and_rows():
    runs = [
        {"run_id": "abc12345", "run_name": "gbm_v1", "metrics": {"accuracy": 0.91}},
        {"run_id": "def67890", "run_name": "rf_v1", "metrics": {"accuracy": 0.87}},
    ]
    table = format_comparison_table(runs, metrics_to_show=["accuracy"])
    assert "run_name" in table
    assert "gbm_v1" in table
    assert "0.9100" in table


def test_table_falls_back_to_run_id_when_unnamed():
    runs = [{"run_id": "abc12345", "run_name": "", "metrics": {"accuracy": 0.5}}]
    table = format_comparison_table(runs, metrics_to_show=["accuracy"])
    assert "abc12345" in table
