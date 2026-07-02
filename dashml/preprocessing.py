"""Config-driven DataFrame cleaning — pure pandas, no Spark/UC dependency."""
from __future__ import annotations
import re
import pandas as pd

from dashml.config import CleaningSpec

_OPS = {
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
    ">": lambda s, v: s > v,
    ">=": lambda s, v: s >= v,
    "<": lambda s, v: s < v,
    "<=": lambda s, v: s <= v,
}


def clean_dataframe(df: pd.DataFrame, spec: CleaningSpec) -> pd.DataFrame:
    """Apply the configured cleaning steps, in order, and return a new DataFrame."""
    if spec.standardize_column_names:
        df = _standardize_column_names(df)

    if spec.drop_nulls_in:
        df = df.dropna(subset=spec.drop_nulls_in)

    if spec.dedup_by:
        df = df.drop_duplicates(subset=[spec.dedup_by])

    for col in spec.date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col, mapping in spec.categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    for derived in spec.derived_columns:
        df = _add_derived_column(df, derived)

    for filt in spec.filters:
        df = _apply_filter(df, filt)

    return df


def _add_derived_column(df: pd.DataFrame, derived) -> pd.DataFrame:
    try:
        values = df.eval(derived.expression)
        df[derived.name] = values.astype(int) if derived.as_binary else values
    except Exception:
        pass  # bad expression — leave the column out rather than fail the whole run
    return df


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"[^\w]", "_", col.strip()).lower() for col in df.columns]
    return df


def _apply_filter(df: pd.DataFrame, filt: dict) -> pd.DataFrame:
    col, op, value = filt.get("column"), filt.get("op", "=="), filt.get("value")
    if not col or col not in df.columns or op not in _OPS:
        return df
    return df[_OPS[op](df[col], value)]
