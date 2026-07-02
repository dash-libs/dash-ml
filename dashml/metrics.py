from __future__ import annotations

from dashml.drift import check_feature_drift, compute_prediction_shift


def compute_drift(baseline_df, production_df, feature_cols: list[str], threshold: float = 0.15) -> dict:
    """Collect the given columns from two Spark DataFrames and compute per-column PSI drift."""
    baseline = {col: _collect_column(baseline_df, col) for col in feature_cols}
    production = {col: _collect_column(production_df, col) for col in feature_cols}
    return check_feature_drift(baseline, production, threshold=threshold)


def compute_prediction_drift(baseline_df, production_df, prediction_col: str) -> dict:
    """Chi-squared test comparing prediction distributions between two Spark DataFrames."""
    chi2, p_value = compute_prediction_shift(
        _collect_column(baseline_df, prediction_col),
        _collect_column(production_df, prediction_col),
    )
    return {"chi2": round(chi2, 4), "p_value": round(p_value, 4), "shifted": p_value < 0.05}


def _collect_column(df, col: str) -> list:
    return [r[col] for r in df.select(col).dropna().collect()]


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
