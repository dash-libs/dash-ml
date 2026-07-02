"""MLflow run tracking + Unity Catalog model registration and promotion.

All MLflow/UC imports are local to functions (never at module level) so this
module is importable without mlflow installed, and doesn't touch Spark.
"""
from __future__ import annotations
from typing import Any


class RunTracker:
    """Thin wrapper around one MLflow run's lifecycle."""

    def __init__(self, experiment_path: str):
        self.experiment_path = experiment_path
        self._own_run = False

    def __enter__(self) -> "RunTracker":
        import mlflow

        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(self.experiment_path)
        active = mlflow.active_run()
        if active is None:
            mlflow.start_run()
            self._own_run = True
        self.run_id = mlflow.active_run().info.run_id
        return self

    def __exit__(self, *exc):
        import mlflow

        if self._own_run:
            mlflow.end_run()

    def log_params(self, params: dict) -> None:
        import mlflow

        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def log_metrics(self, metrics: dict) -> None:
        import mlflow

        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, float(value))
            except (TypeError, ValueError):
                mlflow.log_param(f"metric_{key}", str(value))

    def log_text(self, text: str, artifact_path: str) -> None:
        import mlflow

        mlflow.log_text(text, artifact_path)

    def log_artifact_file(self, path: str, artifact_path: str | None = None) -> None:
        import mlflow

        mlflow.log_artifact(path, artifact_path)


def registered_model_name(catalog: str, schema_name: str, model_name: str) -> str:
    return f"{catalog}.{schema_name}.{model_name}"


def register_model(catalog: str, schema_name: str, model_name: str, model: Any, run_id: str | None = None) -> bool:
    """Register a fitted pyfunc-compatible model into UC via MLflow."""
    import mlflow

    mlflow.set_registry_uri("databricks-uc")
    uc_name = registered_model_name(catalog, schema_name, model_name)
    try:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=_PyfuncWrapper(model),
            registered_model_name=uc_name,
        )
        return True
    except Exception:
        return False


class _PyfuncWrapper:
    """Adapts a plain `.predict()`-style model to the MLflow pyfunc contract."""

    def __init__(self, model: Any):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


def promote_challenger(
    catalog: str,
    schema_name: str,
    model_name: str,
    run_id: str,
    metric_key: str = "accuracy",
    min_threshold: float | None = None,
    direction: str = "max",
    dry_run: bool = False,
) -> str:
    """Compare the given run's metric against the current @champion and, if it wins,
    promote it. Returns one of: promoted, no_champion, retained, below_threshold,
    no_metric, would_promote, no_version, error."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient(registry_uri="databricks-uc")
    uc_name = registered_model_name(catalog, schema_name, model_name)

    run = client.get_run(run_id)
    new_value = run.data.metrics.get(metric_key)
    if new_value is None:
        return "no_metric"
    if min_threshold is not None:
        beats_floor = new_value >= min_threshold if direction == "max" else new_value <= min_threshold
        if not beats_floor:
            return "below_threshold"

    try:
        champion = client.get_model_version_by_alias(uc_name, "champion")
        champion_value = client.get_run(champion.run_id).data.metrics.get(metric_key)
        improved = new_value > champion_value if direction == "max" else new_value < champion_value
        has_champion = True
    except Exception:
        improved, has_champion = True, False

    if not improved:
        return "retained"
    if dry_run:
        return "would_promote"

    versions = client.search_model_versions(f"name='{uc_name}'")
    match = next((v for v in versions if v.run_id == run_id), None)
    if match is None:
        return "no_version"

    try:
        client.set_registered_model_alias(uc_name, "champion", match.version)
        return "promoted" if has_champion else "no_champion"
    except Exception:
        return "error"
