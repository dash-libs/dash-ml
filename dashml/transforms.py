"""Reusable feature-engineering building blocks — pure pandas/numpy functions.

Each takes a DataFrame and returns a new one. Use directly, or chain a few
together inside a custom preprocessing step.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def aggregate_time_window(
    df: pd.DataFrame,
    group_by: str,
    date_col: str = "date",
    window_months: int = 6,
    agg_columns: dict[str, str | list[str]] | None = None,
) -> pd.DataFrame:
    """Restrict to the trailing `window_months` and aggregate per group."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    cutoff = df[date_col].max() - pd.DateOffset(months=window_months)
    df = df[df[date_col] >= cutoff]

    if agg_columns:
        result = df.groupby(group_by).agg(agg_columns).reset_index()
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ["_".join(str(c) for c in col if c).strip("_") for col in result.columns]
    else:
        result = df.groupby(group_by).mean(numeric_only=True).reset_index()
    return result


def aggregate_category_spend(
    df: pd.DataFrame,
    id_col: str,
    category_col: str,
    amount_col: str,
    date_col: str,
    category_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Per-entity average monthly amount per category, pivoted wide, plus an overall average."""
    df = df.copy()
    pivot_col = category_col
    if category_mapping:
        df["_category_group"] = df[category_col].map(category_mapping)
        df = df.dropna(subset=["_category_group"])
        pivot_col = "_category_group"

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["_period"] = df[date_col].dt.to_period("M")

    monthly_by_cat = df.groupby([id_col, "_period", pivot_col])[amount_col].sum().reset_index()
    avg_by_cat = monthly_by_cat.groupby([id_col, pivot_col])[amount_col].mean().reset_index()
    wide = avg_by_cat.pivot(index=id_col, columns=pivot_col, values=amount_col).reset_index()

    monthly_total = df.groupby([id_col, "_period"])[amount_col].sum().reset_index()
    avg_total = monthly_total.groupby(id_col)[amount_col].mean().reset_index()
    avg_total.columns = [id_col, "avg_monthly_amount"]

    return wide.merge(avg_total, on=id_col, how="inner").fillna(0)


def remove_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()

    if method == "zscore":
        from scipy.stats import zscore
        z = np.abs(zscore(df[cols].astype(float), nan_policy="omit"))
        return df[(z < threshold).all(axis=1)]

    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def bin_column(
    df: pd.DataFrame,
    column: str,
    bins: list,
    labels: list,
    output_col: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df[output_col or f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    return df


def encode_categorical(
    df: pd.DataFrame,
    mapping: dict[str, dict] | None = None,
    one_hot_columns: list[str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    for col, col_map in (mapping or {}).items():
        if col in df.columns:
            df[col] = df[col].map(col_map)
    for col in one_hot_columns or []:
        if col in df.columns:
            df = pd.concat([df.drop(columns=[col]), pd.get_dummies(df[col], prefix=col)], axis=1)
    return df


def add_ratio_column(df: pd.DataFrame, numerator: str, denominator: str, name: str) -> pd.DataFrame:
    df = df.copy()
    df[name] = np.where(df[denominator] != 0, df[numerator] / df[denominator], np.nan)
    return df


def add_lag_columns(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
    group_by: str,
    date_col: str = "date",
) -> pd.DataFrame:
    df = df.copy().sort_values([group_by, date_col])
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df.groupby(group_by)[col].shift(lag)
    return df


def impute_nulls(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    strategy: str = "median",
    fill_value: float = 0,
) -> pd.DataFrame:
    df = df.copy()
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        if col not in df.columns:
            continue
        if strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "mode":
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) else fill_value)
        else:
            df[col] = df[col].fillna(fill_value)
    return df
