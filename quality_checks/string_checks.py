import pandas as pd
import numpy as np


def calculate_severity(issue_percentage: float) -> str:
    """
    Determine severity based on the percentage of problematic values.
    """
    if issue_percentage == 0:
        return "none"
    elif issue_percentage <= 5:
        return "low"
    elif issue_percentage <= 20:
        return "medium"
    elif issue_percentage <= 50:
        return "high"
    else:
        return "critical"


def check_empty_strings(df: pd.DataFrame) -> list:
    """
    Detect empty strings in object/string columns.

    Returns:
        A list of structured quality issues.
    """
    issues = []

    for column in df.columns:
        series = df[column]

        # Only check string-like columns
        if not (pd.api.types.is_object_dtype(series) or
                pd.api.types.is_string_dtype(series)):
            continue

        non_null = series.dropna()

        if non_null.empty:
            continue

        # Convert to string and check for empty strings
        string_values = non_null.astype(str)
        empty_mask = string_values == ""

        empty_count = int(empty_mask.sum())

        if empty_count == 0:
            continue

        empty_percentage = (empty_count / len(non_null)) * 100
        severity = calculate_severity(empty_percentage)

        issues.append({
            "issue_type": "empty_strings",
            "column": column,
            "count": empty_count,
            "percentage": round(empty_percentage, 2),
            "severity": severity,
            "evidence": {
                "total_non_null_values": int(len(non_null)),
                "empty_string_count": empty_count,
                "empty_string_percentage": round(empty_percentage, 2),
            }
        })

    return issues


def check_whitespace_only_strings(df: pd.DataFrame) -> list:
    """
    Detect whitespace-only strings in object/string columns.

    Returns:
        A list of structured quality issues.
    """
    issues = []

    for column in df.columns:
        series = df[column]

        # Only check string-like columns
        if not (pd.api.types.is_object_dtype(series) or
                pd.api.types.is_string_dtype(series)):
            continue

        non_null = series.dropna()

        if non_null.empty:
            continue

        # Convert to string and check for whitespace-only strings
        string_values = non_null.astype(str)
        whitespace_mask = string_values.str.strip() == ""

        # Exclude actual empty strings (already caught by empty_strings check)
        whitespace_mask = whitespace_mask & (string_values != "")

        whitespace_count = int(whitespace_mask.sum())

        if whitespace_count == 0:
            continue

        whitespace_percentage = (whitespace_count / len(non_null)) * 100
        severity = calculate_severity(whitespace_percentage)

        issues.append({
            "issue_type": "whitespace_only_strings",
            "column": column,
            "count": whitespace_count,
            "percentage": round(whitespace_percentage, 2),
            "severity": severity,
            "evidence": {
                "total_non_null_values": int(len(non_null)),
                "whitespace_only_count": whitespace_count,
                "whitespace_only_percentage": round(whitespace_percentage, 2),
            }
        })

    return issues