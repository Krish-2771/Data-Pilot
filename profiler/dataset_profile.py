import pandas as pd

from profiler.column_profile import profile_columns
from profiler.statistics import calculate_dataframe_statistics


def calculate_memory_usage(df: pd.DataFrame) -> int:
    """
    Calculate total memory usage of the DataFrame in bytes.
    """

    return int(
        df.memory_usage(deep=True).sum()
    )


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a complete structured profile of a dataset.
    """

    dataframe_statistics = calculate_dataframe_statistics(df)

    return {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "memory_bytes": calculate_memory_usage(df),
        },

        "columns": profile_columns(df),

        "statistics": dataframe_statistics,
    }


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Return names of numeric columns.
    """

    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]


def get_categorical_columns(df: pd.DataFrame) -> list:
    """
    Return names of categorical columns.
    """

    profiles = profile_columns(df)

    return [
        profile["column"]
        for profile in profiles
        if profile["logical_type"] == "categorical"
    ]


def get_datetime_columns(df: pd.DataFrame) -> list:
    """
    Return names of datetime columns.
    """

    profiles = profile_columns(df)

    return [
        profile["column"]
        for profile in profiles
        if profile["logical_type"] == "datetime"
    ]
    
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset.

    Supported formats:
        .csv
        .xlsx
        .xls
    """

    file_path_lower = file_path.lower()

    if file_path_lower.endswith(".csv"):
        return pd.read_csv(file_path)

    if file_path_lower.endswith(".xlsx"):
        return pd.read_excel(file_path)

    if file_path_lower.endswith(".xls"):
        return pd.read_excel(file_path)

    raise ValueError(
        "Unsupported file format. "
        "Only CSV, XLSX and XLS files are supported."
    )


def profile_dataset_file(file_path: str) -> dict:
    """
    Load a dataset from a file and generate
    its complete profile.
    """

    df = load_dataset(file_path)

    return profile_dataset(df)