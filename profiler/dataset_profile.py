import os
import pandas as pd

from profiler.statistics import calculate_dataframe_statistics
from profiler.column_profile import profile_columns


def calculate_memory_usage(df: pd.DataFrame) -> int:
    """
    Calculate total memory usage of a DataFrame in bytes.
    """
    return int(df.memory_usage(deep=True).sum())


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a complete dataset profile.

    Returns:
        Dictionary containing dataset-level statistics and
        per-column profile information.
    """
    statistics = calculate_dataframe_statistics(df)
    column_profiles = profile_columns(df)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "memory_usage_bytes": calculate_memory_usage(df),
        "statistics": statistics,
        "column_profiles": column_profiles,
    }


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Return names of numeric columns.
    """
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    Return names of categorical/object columns.
    """
    return df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()


def get_datetime_columns(df: pd.DataFrame) -> list[str]:
    """
    Return names of datetime columns.
    """
    return df.select_dtypes(include=["datetime"]).columns.tolist()


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset.
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".csv":
        return pd.read_csv(file_path)

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    raise ValueError(
        f"Unsupported file format: {extension}. "
        "Supported formats are CSV and Excel."
    )


def profile_dataset_file(file_path: str) -> dict:
    """
    Load a dataset from a file and generate its profile.
    """
    df = load_dataset(file_path)
    profile = profile_dataset(df)

    profile["file_name"] = os.path.basename(file_path)

    return profile