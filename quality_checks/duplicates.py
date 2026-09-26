import pandas as pd


def calculate_severity(duplicate_percentage: float) -> str:
    if duplicate_percentage == 0:
        return "none"
    elif duplicate_percentage <= 5:
        return "low"
    elif duplicate_percentage <= 20:
        return "medium"
    elif duplicate_percentage <= 50:
        return "high"
    else:
        return "critical"


def check_duplicates(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    total_rows = len(df)

    duplicate_count = int(
        df.duplicated(keep="first").sum()
    )

    if duplicate_count == 0:
        return []

    duplicate_percentage = (
        duplicate_count / total_rows
    ) * 100

    severity = calculate_severity(
        duplicate_percentage
    )

    issue = {
        "issue_type": "duplicate_rows",
        "column": None,
        "count": duplicate_count,
        "percentage": round(
            float(duplicate_percentage),
            2
        ),
        "severity": severity,
        "evidence": {
            "total_rows": total_rows,
            "duplicate_count": duplicate_count,
            "duplicate_percentage": round(
                float(duplicate_percentage),
                2
            )
        }
    }

    return [issue]