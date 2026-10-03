import pandas as pd
import numpy as np
from typing import Literal


def calculate_severity(outlier_percentage: float) -> str:
    """
    Determine severity based on the percentage of outliers.
    """
    if outlier_percentage == 0:
        return "none"
    elif outlier_percentage <= 5:
        return "low"
    elif outlier_percentage <= 20:
        return "medium"
    elif outlier_percentage <= 50:
        return "high"
    else:
        return "critical"


def check_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> list:
    """
    Detect outliers in numerical columns using the IQR method.

    Returns:
        A list of structured quality issues.
    """
    issues = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        # Need at least 4 values for meaningful quartile analysis
        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        # If all values are the same, there cannot be an IQR outlier
        if iqr == 0:
            continue

        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)

        outliers = series[
            (series < lower_bound) |
            (series > upper_bound)
        ]

        outlier_count = int(len(outliers))

        if outlier_count == 0:
            continue

        outlier_percentage = (
            outlier_count / len(series)
        ) * 100

        severity = calculate_severity(
            outlier_percentage
        )

        issues.append({
            "issue_type": "outliers_iqr",
            "column": column,
            "count": outlier_count,
            "percentage": round(
                outlier_percentage, 2
            ),
            "severity": severity,
            "evidence": {
                "total_non_null_values": int(len(series)),
                "outlier_count": outlier_count,
                "outlier_percentage": round(
                    outlier_percentage, 2
                ),
                "q1": round(float(q1), 2),
                "q3": round(float(q3), 2),
                "iqr": round(float(iqr), 2),
                "multiplier": multiplier,
                "lower_bound": round(
                    float(lower_bound), 2
                ),
                "upper_bound": round(
                    float(upper_bound), 2
                )
            }
        })

    return issues


def check_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> list:
    """
    Detect outliers in numerical columns using the Z-score method.

    Returns:
        A list of structured quality issues.
    """
    issues = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        # Need at least 2 values for z-score
        if len(series) < 2:
            continue

        mean = series.mean()
        std = series.std()

        # If std is 0, all values are the same
        if std == 0:
            continue

        z_scores = np.abs((series - mean) / std)
        outliers = series[z_scores > threshold]

        outlier_count = int(len(outliers))

        if outlier_count == 0:
            continue

        outlier_percentage = (
            outlier_count / len(series)
        ) * 100

        severity = calculate_severity(
            outlier_percentage
        )

        issues.append({
            "issue_type": "outliers_zscore",
            "column": column,
            "count": outlier_count,
            "percentage": round(
                outlier_percentage, 2
            ),
            "severity": severity,
            "evidence": {
                "total_non_null_values": int(len(series)),
                "outlier_count": outlier_count,
                "outlier_percentage": round(
                    outlier_percentage, 2
                ),
                "mean": round(float(mean), 2),
                "std": round(float(std), 2),
                "threshold": threshold,
                "max_z_score": round(float(z_scores.max()), 2),
            }
        })

    return issues


def check_outliers(
    df: pd.DataFrame,
    method: Literal["iqr", "zscore"] = "iqr",
    multiplier: float = 1.5,
    threshold: float = 3.0
) -> list:
    """
    Detect outliers in numerical columns using the specified method.

    Args:
        df: Input DataFrame.
        method: "iqr" for IQR method, "zscore" for Z-score method.
        multiplier: IQR multiplier (default 1.5).
        threshold: Z-score threshold (default 3.0).

    Returns:
        A list of structured quality issues.
    """
    if method == "iqr":
        return check_outliers_iqr(df, multiplier)
    elif method == "zscore":
        return check_outliers_zscore(df, threshold)
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")