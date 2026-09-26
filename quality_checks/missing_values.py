import pandas as pd


def calculate_severity(missing_percentage: float) -> str:
    """
    Determine severity based on the percentage of missing values.
    """

    if missing_percentage == 0:
        return "none"

    if missing_percentage <= 5:
        return "low"

    if missing_percentage <= 20:
        return "medium"

    if missing_percentage <= 50:
        return "high"

    return "critical"


def check_missing_values(df: pd.DataFrame) -> list:
    """
    Detect missing values in every column.

    Returns a structured list of quality issues.
    """

    issues = []

    total_rows = len(df)

    if total_rows == 0:
        return issues

    for column in df.columns:

        missing_count = int(df[column].isna().sum())

        if missing_count == 0:
            continue

        missing_percentage = (
            missing_count / total_rows
        ) * 100

        severity = calculate_severity(
            missing_percentage
        )

        issues.append({
            "issue_type": "missing_values",
            "column": column,
            "count": missing_count,
            "percentage": round(
                missing_percentage,
                2
            ),
            "severity": severity,
            "evidence": {
                "total_rows": total_rows,
                "missing_count": missing_count,
                "missing_percentage": round(
                    missing_percentage,
                    2
                )
            }
        })

    return issues