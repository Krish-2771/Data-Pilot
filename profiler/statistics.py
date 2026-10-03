import pandas as pd
import numpy as np
from typing import Optional


def calculate_numeric_statistics(series: pd.Series) -> dict:
    """
    Calculate statistical measures for a numerical column.
    """

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()

    if numeric_series.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "mode": None,
            "minimum": None,
            "maximum": None,
            "range": None,
            "variance": None,
            "standard_deviation": None,
            "q1": None,
            "q3": None,
            "iqr": None,
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
        "count": int(numeric_series.count()),
        "mean": safe_float(numeric_series.mean()),
        "median": safe_float(numeric_series.median()),
        "mode": safe_float(mode_values.iloc[0]) if not mode_values.empty else None,
        "mode_frequency": mode_freq,
        "minimum": safe_float(numeric_series.min()),
        "maximum": safe_float(numeric_series.max()),
        "range": safe_float(numeric_series.max() - numeric_series.min()),
        "variance": safe_float(numeric_series.var()),
        "standard_deviation": safe_float(numeric_series.std()),
        "q1": safe_float(numeric_series.quantile(0.25)),
        "q3": safe_float(numeric_series.quantile(0.75)),
        "iqr": safe_float(
            numeric_series.quantile(0.75)
            - numeric_series.quantile(0.25)
        ),
    }


def calculate_categorical_statistics(series: pd.Series) -> dict:
    """
    Calculate statistics for a categorical column.
    """

    clean_series = series.dropna()

    if clean_series.empty:
        return {
            "count": 0,
            "unique_count": 0,
            "mode": None,
            "mode_frequency": 0,
            "top_values": {},
        }

    value_counts = clean_series.value_counts()

    mode = value_counts.index[0]
    mode_frequency = int(value_counts.iloc[0])
    top_10 = value_counts.head(10)

    return {
        "count": int(clean_series.count()),
        "unique_count": int(clean_series.nunique()),
        "mode": str(mode),
        "mode_frequency": mode_frequency,
        "top_values": {str(k): int(v) for k, v in top_10.items()},
    }


def calculate_datetime_statistics(series: pd.Series) -> dict:
    """
    Calculate statistics for a datetime column.
    """

    dt_series = pd.to_datetime(series, errors="coerce").dropna()

    if dt_series.empty:
        return {
            "count": 0,
            "min_date": None,
            "max_date": None,
            "range_days": None,
        }

    return {
        "count": int(dt_series.count()),
        "min_date": dt_series.min().isoformat(),
        "max_date": dt_series.max().isoformat(),
        "range_days": float((dt_series.max() - dt_series.min()).total_seconds() / 86400),
    }


def calculate_missing_percentage(series: pd.Series) -> float:
    """
    Calculate the percentage of missing values in a column.
    """

    if len(series) == 0:
        return 0.0

    missing_count = int(series.isna().sum())

    return float((missing_count / len(series)) * 100)


def calculate_cardinality(series: pd.Series) -> int:
    """
    Calculate the number of unique non-null values in a column.
    """

    return int(series.dropna().nunique())


def calculate_statistics(series: pd.Series) -> dict:
    """
    Automatically calculate appropriate statistics based
    on the data type of the column.
    """

    result = {
        "column": series.name,
        "data_type": str(series.dtype),
        "missing_count": int(series.isna().sum()),
        "missing_percentage": calculate_missing_percentage(series),
        "cardinality": calculate_cardinality(series),
    }

    # Check boolean first (is_numeric_dtype returns True for bool)
    if pd.api.types.is_bool_dtype(series):
        result["type"] = "boolean"
        result["statistics"] = calculate_categorical_statistics(series)

    elif pd.api.types.is_numeric_dtype(series):
        result["type"] = "numeric"
        result["statistics"] = calculate_numeric_statistics(series)

    elif pd.api.types.is_datetime64_any_dtype(series):
        result["type"] = "datetime"
        result["statistics"] = calculate_datetime_statistics(series)

    else:
        result["type"] = "categorical"
        result["statistics"] = calculate_categorical_statistics(series)

    return result


def calculate_dataframe_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate statistics for every column in a DataFrame.

    Handles duplicate column names by using column positions.
    Returns a list of statistics objects instead of a dict to preserve all columns.
    """

    statistics = []

    for i in range(len(df.columns)):
        column_name = df.columns[i]
        series = df.iloc[:, i]
        # Ensure we have a Series with the correct name
        if isinstance(series, pd.DataFrame):
            # This can happen with duplicate column names
            series = series.iloc[:, 0]
            series.name = column_name

        col_stats = calculate_statistics(series)
        statistics.append(col_stats)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_statistics": statistics,
    }