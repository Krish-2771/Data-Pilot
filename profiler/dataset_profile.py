import os
import pandas as pd
import numpy as np
from typing import Optional

from profiler.statistics import calculate_dataframe_statistics
from profiler.column_profile import profile_columns
from schemas.dataset_schema import (
    DatasetSchema,
    DuplicateRowsSchema,
    OverallMissingSchema,
    ColumnSchema,
)


def calculate_memory_usage(df: pd.DataFrame) -> int:
    """
    Calculate total memory usage of a DataFrame in bytes.
    """
    return int(df.memory_usage(deep=True).sum())


def calculate_duplicate_rows(df: pd.DataFrame) -> dict:
    """
    Calculate duplicate row statistics.
    """
    total_rows = len(df)
    duplicate_count = int(df.duplicated(keep="first").sum())
    duplicate_percentage = (duplicate_count / total_rows * 100) if total_rows > 0 else 0.0

    return {
        "total_rows": total_rows,
        "duplicate_count": duplicate_count,
        "duplicate_percentage": round(duplicate_percentage, 2),
    }


def calculate_overall_missing(df: pd.DataFrame) -> dict:
    """
    Calculate overall missing value statistics for the dataset.
    """
    total_cells = df.size
    missing_cells = int(df.isna().sum().sum())
    missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0.0

    return {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percentage": round(missing_percentage, 2),
    }


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a complete dataset profile.

    Returns:
        Dictionary containing dataset-level statistics and
        per-column profile information.
    """
    statistics = calculate_dataframe_statistics(df)
    column_profiles = profile_columns(df)
    duplicate_info = calculate_duplicate_rows(df)
    missing_info = calculate_overall_missing(df)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "memory_usage_bytes": calculate_memory_usage(df),
        "duplicate_rows": duplicate_info,
        "overall_missing": missing_info,
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


def profile_dataset_as_schema(df: pd.DataFrame) -> DatasetSchema:
    """
    Generate a complete dataset profile as a Pydantic schema object.
    """
    profile_dict = profile_dataset(df)

    # Convert column profiles to ColumnSchema objects
    column_schemas = []
    for col_profile in profile_dict["column_profiles"]:
        column_schemas.append(ColumnSchema(**col_profile))

    return DatasetSchema(
        file_name=profile_dict.get("file_name"),
        rows=profile_dict["rows"],
        columns=profile_dict["columns"],
        memory_usage_bytes=profile_dict["memory_usage_bytes"],
        duplicate_rows=DuplicateRowsSchema(**profile_dict["duplicate_rows"]),
        overall_missing=OverallMissingSchema(**profile_dict["overall_missing"]),
        column_info=column_schemas,
    )