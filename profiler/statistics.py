import pandas as pd
import numpy as np


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
        }

    mode_values = numeric_series.mode()

    return {
        "count": int(numeric_series.count()),
        "mean": float(numeric_series.mean()),
        "median": float(numeric_series.median()),
        "mode": float(mode_values.iloc[0]) if not mode_values.empty else None,
        "minimum": float(numeric_series.min()),
        "maximum": float(numeric_series.max()),
        "range": float(numeric_series.max() - numeric_series.min()),
        "variance": float(numeric_series.var()),
        "standard_deviation": float(numeric_series.std()),
        "q1": float(numeric_series.quantile(0.25)),
        "q3": float(numeric_series.quantile(0.75)),
        "iqr": float(
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
        }

    value_counts = clean_series.value_counts()

    mode = value_counts.index[0]
    mode_frequency = int(value_counts.iloc[0])

    return {
        "count": int(clean_series.count()),
        "unique_count": int(clean_series.nunique()),
        "mode": str(mode),
        "mode_frequency": mode_frequency,
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

    if pd.api.types.is_numeric_dtype(series):
        result["type"] = "numeric"
        result["statistics"] = calculate_numeric_statistics(series)

    else:
        result["type"] = "categorical"
        result["statistics"] = calculate_categorical_statistics(series)

    return result


def calculate_dataframe_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate statistics for every column in a DataFrame.
    """

    statistics = {}

    for column in df.columns:
        statistics[column] = calculate_statistics(df[column])

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_statistics": statistics,
    }