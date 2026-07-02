"""Unit tests for feature-engineering transforms (no Spark required)."""
import numpy as np
import pandas as pd

from dashml.transforms import (
    add_lag_columns,
    add_ratio_column,
    aggregate_time_window,
    bin_column,
    encode_categorical,
    impute_nulls,
    remove_outliers,
)


def test_aggregate_time_window_restricts_to_recent_rows():
    df = pd.DataFrame({
        "id": [1, 1, 2],
        "date": ["2024-01-01", "2024-06-01", "2024-06-01"],
        "value": [100, 10, 20],
    })
    result = aggregate_time_window(df, group_by="id", date_col="date", window_months=1)
    assert set(result["id"]) == {1, 2}
    assert result.loc[result["id"] == 1, "value"].iloc[0] == 10  # Jan row outside window, excluded


def test_remove_outliers_iqr():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
    result = remove_outliers(df, columns=["x"], method="iqr")
    assert 1000 not in result["x"].values


def test_bin_column():
    df = pd.DataFrame({"age": [10, 30, 60]})
    result = bin_column(df, "age", bins=[0, 20, 40, 100], labels=["young", "mid", "senior"])
    assert list(result["age_binned"]) == ["young", "mid", "senior"]


def test_encode_categorical_mapping():
    df = pd.DataFrame({"tier": ["gold", "silver"]})
    result = encode_categorical(df, mapping={"tier": {"gold": 1, "silver": 0}})
    assert list(result["tier"]) == [1, 0]


def test_encode_categorical_one_hot():
    df = pd.DataFrame({"tier": ["gold", "silver"]})
    result = encode_categorical(df, one_hot_columns=["tier"])
    assert "tier_gold" in result.columns and "tier_silver" in result.columns


def test_add_ratio_column_handles_zero_denominator():
    df = pd.DataFrame({"debt": [10, 20], "assets": [5, 0]})
    result = add_ratio_column(df, numerator="debt", denominator="assets", name="ratio")
    assert result["ratio"].iloc[0] == 2.0
    assert np.isnan(result["ratio"].iloc[1])


def test_add_lag_columns():
    df = pd.DataFrame({"id": [1, 1, 1], "date": [1, 2, 3], "value": [10, 20, 30]})
    result = add_lag_columns(df, columns=["value"], lags=[1], group_by="id", date_col="date")
    assert result["value_lag_1"].isna().iloc[0]
    assert list(result["value_lag_1"].iloc[1:]) == [10, 20]


def test_impute_nulls_median():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    result = impute_nulls(df, strategy="median")
    assert result["x"].isna().sum() == 0
    assert result["x"].iloc[1] == 2.0


def test_impute_nulls_constant():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    result = impute_nulls(df, strategy="constant", fill_value=-1)
    assert result["x"].iloc[1] == -1
