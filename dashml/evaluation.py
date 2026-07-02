"""Model evaluation: feature importance (SHAP), threshold gating, model cards.

No Databricks dependency — operates on an already-fitted model and a
feature matrix. MLflow logging, if wanted, is the caller's job.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd


def check_thresholds(metrics: dict[str, float], min_thresholds: dict[str, float]) -> dict[str, bool]:
    """Per-metric pass/fail against configured minimums."""
    return {name: metrics.get(name, float("-inf")) >= minimum for name, minimum in min_thresholds.items()}


def explain_features(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    sample_size: int = 100,
    background_size: int = 50,
) -> dict[str, float] | None:
    """Mean absolute SHAP value per feature. Works with any model exposing
    predict() or scikit-learn-style kneighbors(). Returns None if `shap` isn't
    installed, the model exposes neither method, or there are too few rows."""
    try:
        import shap
    except ImportError:
        return None

    if isinstance(X, pd.DataFrame):
        feature_names = feature_names or list(X.columns)
        X_numeric = X.copy()
        for col in X_numeric.columns:
            if not pd.api.types.is_numeric_dtype(X_numeric[col]):
                X_numeric[col] = pd.factorize(X_numeric[col])[0]
        X_arr = X_numeric.to_numpy(dtype=np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        feature_names = feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])]

    X_arr = np.nan_to_num(X_arr)
    if len(X_arr) < 5:
        return None

    predict_fn = _predict_function(model, len(X_arr))
    if predict_fn is None:
        return None

    try:
        background = shap.sample(X_arr, min(background_size, len(X_arr)))
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_arr[: min(sample_size, len(X_arr))])
        return dict(zip(feature_names, np.abs(shap_values).mean(axis=0).tolist()))
    except Exception:
        return None


def _predict_function(model: Any, n_rows: int):
    if hasattr(model, "predict"):
        return lambda x: model.predict(x)
    if hasattr(model, "kneighbors"):
        k = min(getattr(model, "n_neighbors", 5), max(getattr(model, "n_samples_fit_", n_rows) - 1, 1))
        if k < 1:
            return None
        return lambda x: model.kneighbors(x, n_neighbors=k)[0].mean(axis=1)
    return None


def plot_feature_importance(shap_by_group: dict[str, dict[str, float]], output_dir: str | None = None) -> list[str]:
    """One horizontal-bar PNG per group in shap_by_group. Returns saved file paths."""
    import os
    import tempfile
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = output_dir or tempfile.mkdtemp()
    paths = []
    for group, importances in shap_by_group.items():
        if not importances:
            continue
        top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:15]
        names, values = [t[0] for t in top], [t[1] for t in top]

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
        ax.barh(range(len(names)), values, color="#2A9D90")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Feature importance — {group}")
        plt.tight_layout()

        safe_name = "".join(c if c.isalnum() else "_" for c in group)[:30]
        path = os.path.join(output_dir, f"importance_{safe_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def build_model_card(
    name: str,
    version: str,
    model_type: str,
    catalog: str,
    schema_name: str,
    tables: list[str],
    target_column: str | None,
    feature_columns: list[str],
    params: dict,
    metrics: dict,
    run_id: str | None = None,
    shap_by_group: dict[str, dict[str, float]] | None = None,
) -> str:
    """Render a Markdown model card. Doesn't write or log it — caller decides where it goes."""
    lines = [
        f"# Model Card — {name}",
        "",
        "## Details",
        f"- **Version:** {version}",
        f"- **Type:** {model_type}",
        f"- **Created:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    if run_id:
        lines.append(f"- **MLflow run:** `{run_id}`")

    lines += [
        "",
        "## Data",
        f"- **Source:** Unity Catalog `{catalog}.{schema_name}`",
        f"- **Tables:** {', '.join(tables) if tables else '—'}",
    ]
    if target_column:
        lines.append(f"- **Target column:** `{target_column}`")
    if feature_columns:
        lines.append(f"- **Feature columns:** {', '.join(feature_columns)}")

    lines += ["", "## Parameters"]
    lines += [f"- `{k}`: {v}" for k, v in sorted(params.items())] or ["_none logged_"]

    lines += ["", "## Metrics"]
    lines += [f"- `{k}`: {v}" for k, v in sorted(metrics.items())] or ["_none logged_"]

    if shap_by_group:
        lines += ["", "## Feature importance (SHAP)"]
        for group, importances in shap_by_group.items():
            top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
            lines.append(f"**{group}**")
            lines += [f"  - `{feat}`: {val:.4f}" for feat, val in top]

    lines += [
        "",
        "## Limitations",
        "Evaluated on the configured evaluation split only — verify performance holds",
        "on your actual production population before relying on this model.",
    ]
    return "\n".join(lines)
