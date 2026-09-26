import pandas as pd


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


def check_outliers(df: pd.DataFrame) -> list:
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

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

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
            "issue_type": "outliers",
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
                "lower_bound": round(
                    float(lower_bound), 2
                ),
                "upper_bound": round(
                    float(upper_bound), 2
                )
            }
        })

    return issues