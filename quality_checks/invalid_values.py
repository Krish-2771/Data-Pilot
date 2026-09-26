import pandas as pd


def calculate_severity(invalid_percentage: float) -> str:
    """
    Determine severity based on the percentage
    of invalid values.
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


def check_invalid_values(df: pd.DataFrame) -> list:
    """
    Detect logically invalid values in a DataFrame.

    Checks currently included:
    - Age values outside 0-120
    - Negative numeric values

    Returns:
        A list of structured quality issues.
    """

    issues = []

    for column in df.columns:
        series = df[column]

        non_null = series.dropna()

        if non_null.empty:
            continue

        # -------------------------------------------------
        # Check 1: Invalid age values
        # -------------------------------------------------
        if column.lower() == "age":
            if not pd.api.types.is_numeric_dtype(series):
                continue

            invalid = series[
                (series < 0) |
                (series > 120)
            ].dropna()

            invalid_count = int(len(invalid))

            if invalid_count > 0:
                invalid_percentage = (
                    invalid_count / len(non_null)
                ) * 100

                severity = calculate_severity(
                    invalid_percentage
                )

                issues.append({
                    "issue_type": "invalid_age_values",
                    "column": column,
                    "count": invalid_count,
                    "percentage": round(
                        invalid_percentage,
                        2
                    ),
                    "severity": severity,
                    "evidence": {
                        "total_non_null_values": int(
                            len(non_null)
                        ),
                        "invalid_value_count": invalid_count,
                        "invalid_percentage": round(
                            invalid_percentage,
                            2
                        ),
                        "valid_range": {
                            "minimum": 0,
                            "maximum": 120
                        }
                    }
                })

                continue

        # -------------------------------------------------
        # Check 2: Negative numeric values
        # -------------------------------------------------
        if pd.api.types.is_numeric_dtype(series):
            invalid = series[
                series < 0
            ].dropna()

            invalid_count = int(len(invalid))

            if invalid_count > 0:
                invalid_percentage = (
                    invalid_count / len(non_null)
                ) * 100

                severity = calculate_severity(
                    invalid_percentage
                )

                issues.append({
                    "issue_type": "negative_numeric_values",
                    "column": column,
                    "count": invalid_count,
                    "percentage": round(
                        invalid_percentage,
                        2
                    ),
                    "severity": severity,
                    "evidence": {
                        "total_non_null_values": int(
                            len(non_null)
                        ),
                        "invalid_value_count": invalid_count,
                        "invalid_percentage": round(
                            invalid_percentage,
                            2
                        ),
                        "valid_rule": "value must be >= 0"
                    }
                })

    return issues