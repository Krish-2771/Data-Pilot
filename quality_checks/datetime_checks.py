import pandas as pd
import numpy as np


def calculate_severity(invalid_percentage: float) -> str:
    """
    Determine severity based on the percentage of invalid datetime values.
    """
    if invalid_percentage == 0:
        return "none"
    elif invalid_percentage <= 5:
        return "low"
    elif invalid_percentage <= 20:
        return "medium"
    elif invalid_percentage <= 50:
        return "high"
    else:
        return "critical"


def check_invalid_datetimes(df: pd.DataFrame) -> list:
    """
    Detect invalid datetime values in columns that appear to be datetime-like.

    Returns:
        A list of structured quality issues.
    """
    issues = []

    for column in df.columns:
        series = df[column]

        non_null = series.dropna()

        if non_null.empty:
            continue

        # Skip if already datetime type
        if pd.api.types.is_datetime64_any_dtype(series):
            # Check for NaT (Not a Time) values
            nat_count = int(series.isna().sum())
            if nat_count > 0:
                nat_percentage = (nat_count / len(series)) * 100
                severity = calculate_severity(nat_percentage)

                issues.append({
                    "issue_type": "invalid_datetime_values",
                    "column": column,
                    "count": nat_count,
                    "percentage": round(nat_percentage, 2),
                    "severity": severity,
                    "evidence": {
                        "total_values": int(len(series)),
                        "nat_count": nat_count,
                        "nat_percentage": round(nat_percentage, 2),
                        "reason": "NaT (Not a Time) values present"
                    }
                })
            continue

        # For non-datetime columns, try to parse as datetime
        string_values = non_null.astype(str)
        datetime_conversion = pd.to_datetime(
            string_values,
            errors="coerce",
            format="mixed"
        )

        # Only check columns where at least some values could be datetime
        datetime_count = int(datetime_conversion.notna().sum())
        if datetime_count == 0:
            continue

        datetime_percentage = (datetime_count / len(string_values)) * 100

        # If most values are datetime-like, check for invalid ones
        if datetime_percentage >= 50:
            nat_count = int(datetime_conversion.isna().sum())
            if nat_count > 0:
                invalid_percentage = (nat_count / len(string_values)) * 100
                severity = calculate_severity(invalid_percentage)

                issues.append({
                    "issue_type": "invalid_datetime_values",
                    "column": column,
                    "count": nat_count,
                    "percentage": round(invalid_percentage, 2),
                    "severity": severity,
                    "evidence": {
                        "total_non_null_values": int(len(string_values)),
                        "datetime_like_count": datetime_count,
                        "invalid_count": nat_count,
                        "invalid_percentage": round(invalid_percentage, 2),
                    }
                })

    return issues