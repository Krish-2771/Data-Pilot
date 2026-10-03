import pandas as pd
import numpy as np
from typing import Optional, Union


def detect_column_type(series: pd.Series) -> str:
    """
    Detect the logical type of a DataFrame column.

    Returns:
        numeric
        categorical
        text
        datetime
        boolean
        unknown
    """

    # Boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Remove missing values for further inspection
    non_null = series.dropna()

    if non_null.empty:
        return "unknown"

    # Try detecting datetime values
    converted_datetime = pd.to_datetime(
        non_null,
        errors="coerce",
        format="mixed"
    )

    datetime_ratio = converted_datetime.notna().mean()

    if datetime_ratio >= 0.8:
        return "datetime"

    # Convert values to strings for categorical/text analysis
    string_values = non_null.astype(str)

    unique_count = string_values.nunique()
    total_count = len(string_values)

    unique_ratio = unique_count / total_count

    # Categorical:
    # relatively small number of unique values
    if unique_count <= 20 or unique_ratio <= 0.05:
        return "categorical"

    # Otherwise treat as text
    return "text"


def _get_numeric_stats(series: pd.Series) -> dict:
    """Calculate detailed statistics for numeric columns."""
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()

    if numeric_series.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "variance": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "mode": None,
            "mode_frequency": None,
        }

    mode_values = numeric_series.mode()

    # Calculate mode frequency properly using value_counts
    value_counts = numeric_series.value_counts()
    mode_freq = int(value_counts.iloc[0]) if not value_counts.empty else None

    # Helper to safely convert to float, handling inf/nan
    def safe_float(val):
        try:
            f = float(val)
            if np.isinf(f) or np.isnan(f):
                return None
            return f
        except (ValueError, OverflowError, TypeError):
            return None

    return {
        "min": safe_float(numeric_series.min()),
        "max": safe_float(numeric_series.max()),
        "mean": safe_float(numeric_series.mean()),
        "median": safe_float(numeric_series.median()),
        "std": safe_float(numeric_series.std()),
        "variance": safe_float(numeric_series.var()),
        "q1": safe_float(numeric_series.quantile(0.25)),
        "q3": safe_float(numeric_series.quantile(0.75)),
        "iqr": safe_float(numeric_series.quantile(0.75) - numeric_series.quantile(0.25)),
        "mode": safe_float(mode_values.iloc[0]) if not mode_values.empty else None,
        "mode_frequency": mode_freq,
    }


def _get_categorical_stats(series: pd.Series) -> dict:
    """Calculate detailed statistics for categorical/text/boolean columns."""
    clean_series = series.dropna()

    if clean_series.empty:
        return {
            "mode": None,
            "mode_frequency": None,
            "top_values": {},
        }

    value_counts = clean_series.value_counts()
    top_10 = value_counts.head(10)

    return {
        "mode": str(value_counts.index[0]),
        "mode_frequency": int(value_counts.iloc[0]),
        "top_values": {str(k): int(v) for k, v in top_10.items()},
    }


def _get_datetime_stats(series: pd.Series) -> dict:
    """Calculate detailed statistics for datetime columns."""
    dt_series = pd.to_datetime(series, errors="coerce").dropna()

    if dt_series.empty:
        return {
            "min_date": None,
            "max_date": None,
            "range_days": None,
        }

    return {
        "min_date": dt_series.min().isoformat(),
        "max_date": dt_series.max().isoformat(),
        "range_days": float((dt_series.max() - dt_series.min()).total_seconds() / 86400),
    }


def profile_column(series: pd.Series) -> dict:
    """
    Generate a structured profile for a single column.

    The returned field names match ColumnSchema in
    schemas/dataset_schema.py.
    """

    column_type = detect_column_type(series)

    total_values = int(len(series))
    non_null_count = int(series.notna().sum())
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))

    missing_percentage = (
        (missing_count / total_values) * 100
        if total_values > 0
        else 0.0
    )

    # Base profile
    profile = {
        "name": str(series.name),
        "data_type": str(series.dtype),
        "column_type": column_type,
        "non_null_count": non_null_count,
        "missing_count": missing_count,
        "missing_percentage": round(float(missing_percentage), 2),
        "unique_count": unique_count,
    }

    # Add type-specific statistics
    if column_type == "numeric":
        profile["statistics"] = _get_numeric_stats(series)
    elif column_type == "datetime":
        profile["statistics"] = _get_datetime_stats(series)
    else:
        profile["statistics"] = _get_categorical_stats(series)

    return profile


def profile_columns(df: pd.DataFrame) -> list:
    """
    Generate profiles for all columns in a DataFrame.

    Handles duplicate column names by profiling each column position separately.
    """

    profiles = []

    for i in range(len(df.columns)):
        column_name = df.columns[i]
        series = df.iloc[:, i]
        # Ensure we have a Series with the correct name
        if isinstance(series, pd.DataFrame):
            # This can happen with duplicate column names
            series = series.iloc[:, 0]
            series.name = column_name
        profiles.append(
            profile_column(series)
        )

    return profiles


def get_columns_by_type(
    df: pd.DataFrame,
    column_type: str
) -> list:
    """
    Return column names belonging to a particular logical type.

    Handles duplicate column names by checking each column position.
    Returns column names (with duplicates if multiple columns have the same name and type).
    """

    columns = []

    for i in range(len(df.columns)):
        column_name = df.columns[i]
        series = df.iloc[:, i]
        # Ensure we have a Series with the correct name
        if isinstance(series, pd.DataFrame):
            # This can happen with duplicate column names
            series = series.iloc[:, 0]
            series.name = column_name

        detected_type = detect_column_type(series)

        if detected_type == column_type:
            columns.append(column_name)

    return columns