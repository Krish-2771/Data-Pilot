import pandas as pd
import pytest

from utils.validators import (
    validate_dataframe,
    validate_columns,
    validate_numeric_columns,
    validate_categorical_columns,
)


def create_sample_dataframe():
    return pd.DataFrame({
        "Age": [21, 22, 25],
        "Salary": [35000, 42000, 55000],
        "Department": ["IT", "HR", "Finance"],
    })


def test_validate_dataframe():
    df = create_sample_dataframe()

    validate_dataframe(df)


def test_validate_dataframe_invalid_input():
    with pytest.raises(TypeError):
        validate_dataframe([1, 2, 3])


def test_validate_columns():
    df = create_sample_dataframe()

    validate_columns(df, ["Age", "Salary"])


def test_validate_columns_invalid():
    df = create_sample_dataframe()

    with pytest.raises(ValueError):
        validate_columns(df, ["Age", "Unknown"])


def test_validate_numeric_columns():
    df = create_sample_dataframe()

    validate_numeric_columns(df, ["Age", "Salary"])


def test_validate_numeric_columns_invalid():
    df = create_sample_dataframe()

    with pytest.raises(ValueError):
        validate_numeric_columns(df, ["Department"])


def test_validate_categorical_columns():
    df = create_sample_dataframe()

    validate_categorical_columns(df, ["Department"])


def test_validate_categorical_columns_invalid():
    df = create_sample_dataframe()

    with pytest.raises(ValueError):
        validate_categorical_columns(df, ["Age"])