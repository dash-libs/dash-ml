"""Unit tests for clean_dataframe (no Spark required)."""
import pandas as pd

from dashml.config import CleaningSpec, DerivedColumn
from dashml.preprocessing import clean_dataframe


def _df():
    return pd.DataFrame({
        "Customer ID": [1, 2, 2, 3, None],
        "Signup Date": ["2024-01-01", "2024-02-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "segment": ["gold", "silver", "silver", None, "gold"],
    })


def test_standardizes_column_names():
    result = clean_dataframe(_df(), CleaningSpec())
    assert "customer_id" in result.columns
    assert "signup_date" in result.columns


def test_dedup_by():
    # standardize_column_names runs first, so dedup_by/drop_nulls_in/date_columns
    # must reference the already-standardized name.
    result = clean_dataframe(_df(), CleaningSpec(dedup_by="customer_id"))
    assert len(result) == 4  # one duplicate id=2 row removed, null id untouched


def test_drop_nulls_in():
    result = clean_dataframe(_df(), CleaningSpec(drop_nulls_in=["customer_id"]))
    assert result["customer_id"].isna().sum() == 0


def test_date_column_parsed():
    result = clean_dataframe(_df(), CleaningSpec(date_columns=["signup_date"]))
    assert pd.api.types.is_datetime64_any_dtype(result["signup_date"])


def test_categorical_mapping():
    spec = CleaningSpec(categorical_mappings={"segment": {"gold": 1, "silver": 0}})
    result = clean_dataframe(_df(), spec)
    assert set(result["segment"].dropna().unique()) <= {0, 1}


def test_derived_column_numeric():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    spec = CleaningSpec(standardize_column_names=False, derived_columns=[DerivedColumn(name="total", expression="a + b")])
    result = clean_dataframe(df, spec)
    assert list(result["total"]) == [11, 22, 33]


def test_derived_column_binary():
    df = pd.DataFrame({"a": [1, 2, 3]})
    spec = CleaningSpec(standardize_column_names=False, derived_columns=[DerivedColumn(name="is_big", expression="a > 1", as_binary=True)])
    result = clean_dataframe(df, spec)
    assert list(result["is_big"]) == [0, 1, 1]


def test_bad_derived_expression_is_skipped_not_raised():
    df = pd.DataFrame({"a": [1, 2, 3]})
    spec = CleaningSpec(standardize_column_names=False, derived_columns=[DerivedColumn(name="bad", expression="not_a_column * 2")])
    result = clean_dataframe(df, spec)
    assert "bad" not in result.columns


def test_filters():
    df = pd.DataFrame({"amount": [10, 50, 100]})
    spec = CleaningSpec(standardize_column_names=False, filters=[{"column": "amount", "op": ">", "value": 20}])
    result = clean_dataframe(df, spec)
    assert list(result["amount"]) == [50, 100]
