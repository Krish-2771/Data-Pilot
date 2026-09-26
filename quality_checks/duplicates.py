import pandas as pd


def calculate_severity(duplicate_percentage: float) -> str:
    """
    Determine severity based on the percentage of duplicate rows.
    """

    if duplicate_percentage == 0:
        return "none"

    if duplicate_percentage <= 5:
        return "low"

    if duplicate_percentage <= 20:
        return "medium"

    if duplicate_percentage <= 50:
        return "high"

    return "critical"


def check_duplicates(df: pd.DataFrame) -> dict:
    """
    Detect duplicate rows in a DataFrame.

    Returns a structured quality result.
    """

    total_rows = len(df)

    if total_rows == 0:
        return {
            "issue_type": "duplicates",
            "count": 0,
            "percentage": 0.0,
            "severity": "none",
            "evidence": {
                "total_rows": 0,
                "duplicate_rows": 0,
            }
        }

    duplicate_mask = df.duplicated(keep="first")

    duplicate_count = int(duplicate_mask.sum())

    duplicate_percentage = (
        duplicate_count / total_rows
    ) * 100

    severity = calculate_severity(
        duplicate_percentage
    )

    return {
        "issue_type": "duplicates",
        "count": duplicate_count,
        "percentage": round(
            duplicate_percentage,
            2
        ),
        "severity": severity,
        "evidence": {
            "total_rows": total_rows,
            "duplicate_rows": duplicate_count,
            "unique_rows": int(
                df.drop_duplicates().shape[0]
            )
        }
    }