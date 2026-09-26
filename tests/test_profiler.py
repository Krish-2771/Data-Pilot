import pandas as pd

from profiler.column_profile import (
    detect_column_type,
    profile_column,
    profile_columns,
)
from profiler.dataset_profile import (
    profile_dataset,
    load_dataset,
)


def create_sample_dataframe():
    return pd.DataFrame({
        "Name": ["Rahul", "Amit", "Priya", "Neha"],
        "Age": [21, 22, None, 25],
        "Department": ["IT", "HR", "IT", "Finance"],
    })


def test_detect_column_type_numeric():
    series = pd.Series([10, 20, 30, 40])

    assert detect_column_type(series) == "numeric"


def test_detect_column_type_categorical():
    series = pd.Series(["IT", "HR", "IT", "Finance"])

    assert detect_column_type(series) == "categorical"


def test_detect_column_type_boolean():
    series = pd.Series([True, False, True, False])

    assert detect_column_type(series) == "boolean"


def test_detect_column_type_datetime():
    series = pd.Series([
        "2026-01-01",
        "2026-02-01",
        "2026-03-01",
        "2026-04-01",
    ])

    assert detect_column_type(series) == "datetime"


def test_profile_column():
    series = pd.Series(
        [21, 22, None, 25],
        name="Age"
    )

    profile = profile_column(series)

    assert profile["name"] == "Age"
    assert profile["data_type"] == "float64"
    assert profile["column_type"] == "numeric"
    assert profile["non_null_count"] == 3
    assert profile["missing_count"] == 1
    assert profile["missing_percentage"] == 25.0
    assert profile["unique_count"] == 3


def test_profile_columns():
    df = create_sample_dataframe()

    profiles = profile_columns(df)

    assert len(profiles) == 3

    column_names = [
        profile["name"]
        for profile in profiles
    ]

    assert column_names == [
        "Name",
        "Age",
        "Department",
    ]


def test_profile_dataset():
    df = create_sample_dataframe()

    profile = profile_dataset(df)

    assert profile["rows"] == 4
    assert profile["columns"] == 3
    assert profile["memory_usage_bytes"] > 0
    assert len(profile["column_profiles"]) == 3


def test_load_csv_dataset():
    df = load_dataset(
        "data/sample/sample_dataset.csv"
    )

    assert len(df) == 10
    assert len(df.columns) == 5
    assert list(df.columns) == [
        "Name",
        "Age",
        "Salary",
        "Department",
        "City",
    ]