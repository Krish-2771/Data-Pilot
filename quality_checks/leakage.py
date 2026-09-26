import pandas as pd


LEAKAGE_NAME_PATTERNS = [
    "target",
    "label",
    "outcome",
    "prediction",
    "predicted",
    "future",
    "result",
    "post_",
    "after_"
]


def calculate_severity(leakage_percentage: float) -> str:
    """
    Determine severity based on the percentage of
    non-null values in a potential leakage column.
    """
    if leakage_percentage <= 20:
        return "low"
    elif leakage_percentage <= 50:
        return "medium"
    elif leakage_percentage <= 80:
        return "high"
    else:
        return "critical"


def has_leakage_like_name(column: str) -> bool:
    """
    Check whether a column name contains terms that
    may indicate target or future information.
    """
    column_name = str(column).lower().strip()

    for pattern in LEAKAGE_NAME_PATTERNS:
        if pattern in column_name:
            return True

    return False


def check_leakage(df: pd.DataFrame) -> list:
    """
    Detect potential data leakage based on column names
    and suspicious relationships with other columns.

    Current detection signals:
    - Leakage-like column names
    - Columns that are exact duplicates of another column

    Returns:
        A list of structured quality issues.
    """

    issues = []

    if df.empty:
        return issues

    # --------------------------------------------------
    # Check 1: Leakage-like column names
    # --------------------------------------------------

    for column in df.columns:
        series = df[column]

        non_null_count = int(series.notna().sum())

        if non_null_count == 0:
            continue

        total_rows = len(df)

        non_null_percentage = (
            non_null_count / total_rows
        ) * 100

        if not has_leakage_like_name(column):
            continue

        severity = calculate_severity(
            non_null_percentage
        )

        issues.append({
            "issue_type": "possible_data_leakage",
            "column": column,
            "count": non_null_count,
            "percentage": round(
                float(non_null_percentage),
                2
            ),
            "severity": severity,
            "evidence": {
                "total_rows": int(total_rows),
                "non_null_count": non_null_count,
                "non_null_percentage": round(
                    float(non_null_percentage),
                    2
                ),
                "reason": "leakage_like_column_name"
            }
        })

    # --------------------------------------------------
    # Check 2: Exact duplicate columns
    # --------------------------------------------------

    columns = list(df.columns)

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            column_1 = columns[i]
            column_2 = columns[j]

            series_1 = df[column_1]
            series_2 = df[column_2]

            if series_1.equals(series_2):
                total_rows = len(df)

                issues.append({
                    "issue_type": "duplicate_information",
                    "column": column_1,
                    "count": total_rows,
                    "percentage": 100.0,
                    "severity": "high",
                    "evidence": {
                        "column_1": column_1,
                        "column_2": column_2,
                        "matching_rows": total_rows,
                        "reason": "exact_duplicate_columns"
                    }
                })

    return issues