"""Governance artifact builders — model documentation that travels with the
registered model, not just the MLflow run.

Sensitive/PII columns aren't guessed at with a hardcoded domain list; pass
what applies to your data via `sensitive_columns`/`pii_columns`.
"""
from __future__ import annotations
import json
import yaml


def build_governance_artifacts(
    name: str,
    version: str,
    catalog: str,
    schema_name: str,
    tables: list[str],
    feature_columns: list[str],
    metrics: dict,
    signature: dict | None = None,
    sensitive_columns: list[str] | None = None,
    pii_columns: list[str] | None = None,
    fairness_metrics: dict | None = None,
    baseline_metrics: dict | None = None,
    per_segment_metrics: dict | None = None,
    shap_by_group: dict[str, dict[str, float]] | None = None,
    required_approvers: list[str] | None = None,
    checklist: list[str] | None = None,
) -> dict[str, str]:
    """Returns {filename: content} for the standard governance file set."""
    files = {
        "signature.json": json.dumps(signature or {}, indent=2),
        "features.json": json.dumps(
            {
                "feature_columns": feature_columns,
                "sensitive_columns": sensitive_columns or [],
                "pii_columns": pii_columns or [],
            },
            indent=2,
        ),
        "metrics.json": json.dumps(
            {
                "metrics": metrics,
                "per_segment_metrics": per_segment_metrics or {},
                "baseline_comparison": _delta(metrics, baseline_metrics) if baseline_metrics else None,
                "top_features": _top_shap(shap_by_group),
            },
            indent=2,
        ),
        "data_sources.yaml": yaml.dump(
            {"catalog": catalog, "schema": schema_name, "tables": [f"{catalog}.{schema_name}.{t}" for t in tables]},
            sort_keys=False,
        ),
        "fairness_report.md": _fairness_report(sensitive_columns, fairness_metrics),
        "approval_record.json": json.dumps(
            {
                "model": name,
                "version": version,
                "state": "pending",
                "required_approvers": required_approvers or [],
                "checklist": {item: False for item in (checklist or [])},
            },
            indent=2,
        ),
    }
    return files


def _delta(metrics: dict, baseline: dict) -> dict:
    return {
        k: {"current": v, "baseline": baseline.get(k), "delta": round(v - baseline[k], 4)}
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and k in baseline
    }


def _top_shap(shap_by_group: dict[str, dict[str, float]] | None, top_n: int = 5) -> dict:
    if not shap_by_group:
        return {}
    return {
        group: [f for f, _ in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
        for group, importances in shap_by_group.items()
    }


def _fairness_report(sensitive_columns: list[str] | None, fairness_metrics: dict | None) -> str:
    lines = ["# Fairness Report", ""]
    lines.append(f"**Excluded sensitive columns:** {', '.join(sensitive_columns) if sensitive_columns else 'none configured'}")
    lines.append("")
    if not fairness_metrics:
        lines.append("No fairness metrics supplied for this run.")
        return "\n".join(lines)

    lines.append("| Group | Metric | Value | Flag |")
    lines.append("|---|---|---|---|")
    for group, group_metrics in fairness_metrics.items():
        for metric, value in group_metrics.items():
            flag = "⚠️ review" if (isinstance(value, (int, float)) and abs(value) > 0.1) else ""
            lines.append(f"| {group} | {metric} | {value} | {flag} |")
    return "\n".join(lines)
