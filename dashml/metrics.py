from __future__ import annotations
from pyspark.sql import functions as F


def compute_drift(baseline_df, production_df, feature_cols: list[str]) -> dict:
    """Compute Population Stability Index (PSI) for numeric features."""
    results = {}
    for col in feature_cols:
        try:
            psi = _psi(baseline_df, production_df, col)
            status = "stable" if psi < 0.1 else ("moderate" if psi < 0.25 else "high_drift")
            results[col] = {"psi": round(psi, 4), "status": status}
        except Exception as e:
            results[col] = {"psi": None, "error": str(e)}
    return results


def _psi(base_df, prod_df, col: str, bins: int = 10) -> float:
    import numpy as np
    base_vals = [r[col] for r in base_df.select(col).dropna().collect()]
    prod_vals = [r[col] for r in prod_df.select(col).dropna().collect()]
    if not base_vals or not prod_vals:
        return 0.0
    _, edges = np.histogram(base_vals, bins=bins)
    base_counts, _ = np.histogram(base_vals, bins=edges)
    prod_counts, _ = np.histogram(prod_vals, bins=edges)
    base_pct = base_counts / len(base_vals)
    prod_pct = prod_counts / len(prod_vals)
    base_pct = np.where(base_pct == 0, 1e-6, base_pct)
    prod_pct = np.where(prod_pct == 0, 1e-6, prod_pct)
    return float(np.sum((prod_pct - base_pct) * np.log(prod_pct / base_pct)))


def compute_performance(df, target_col: str, pred_col: str) -> dict:
    """Compute accuracy and error metrics."""
    from pyspark.sql import functions as F
    total = df.count()
    correct = df.filter(F.col(target_col) == F.col(pred_col)).count()
    accuracy = correct / total if total > 0 else 0.0

    mae_row = df.agg(F.mean(F.abs(F.col(pred_col).cast("double") -
                                  F.col(target_col).cast("double"))).alias("mae")).collect()[0]
    return {
        "accuracy": round(accuracy, 4),
        "mae": round(float(mae_row["mae"] or 0), 4),
        "total_predictions": total,
    }
