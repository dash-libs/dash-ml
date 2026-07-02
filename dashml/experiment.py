"""Compare and promote MLflow runs logged under one experiment.

Unlike a CLI-driven workflow with a local project directory, there's no
on-disk snapshot of "which code produced which run" here — everything is
read back from MLflow itself, which already has the run history.
"""
from __future__ import annotations


def compare_runs(
    experiment_path: str,
    metrics_to_compare: list[str] | None = None,
    sort_by: str | None = None,
    descending: bool = True,
) -> list[dict]:
    """List runs under `experiment_path` with their metrics, most relevant metric first."""
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    experiment = mlflow.get_experiment_by_name(experiment_path)
    if experiment is None:
        return []

    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs_df.empty:
        return []

    metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
    if metrics_to_compare:
        metric_cols = [c for c in metric_cols if c.removeprefix("metrics.") in metrics_to_compare]

    results = []
    for _, row in runs_df.iterrows():
        results.append(
            {
                "run_id": row["run_id"],
                "run_name": row.get("tags.mlflow.runName", ""),
                "start_time": row.get("start_time"),
                "metrics": {c.removeprefix("metrics."): row[c] for c in metric_cols if row[c] is not None},
            }
        )

    sort_key = sort_by or (metrics_to_compare[0] if metrics_to_compare else None)
    if sort_key:
        results.sort(key=lambda r: r["metrics"].get(sort_key, float("-inf")), reverse=descending)
    return results


def format_comparison_table(runs: list[dict], metrics_to_show: list[str] | None = None) -> str:
    if not runs:
        return "No runs found."

    columns = metrics_to_show or sorted({k for r in runs for k in r["metrics"]})
    header = ["run_name"] + columns
    rows = [[r["run_name"] or r["run_id"][:8]] + [f"{r['metrics'].get(c, '—'):.4f}" if isinstance(r["metrics"].get(c), (int, float)) else "—" for c in columns] for r in runs]

    widths = [max(len(str(x)) for x in [header[i], *[row[i] for row in rows]]) for i in range(len(header))]
    lines = [" | ".join(h.ljust(w) for h, w in zip(header, widths))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    return "\n".join(lines)


def promote_run(catalog: str, schema_name: str, model_name: str, run_id: str) -> dict:
    """Register a run's logged model artifact into UC as a new version."""
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    uc_name = f"{catalog}.{schema_name}.{model_name}"
    try:
        result = mlflow.register_model(f"runs:/{run_id}/model", uc_name)
        return {"status": "registered", "name": uc_name, "version": result.version}
    except Exception as e:
        return {"status": "error", "name": uc_name, "error": str(e)}
