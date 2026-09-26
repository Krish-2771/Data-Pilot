import pandas as pd


def calculate_severity(dominant_percentage: float) -> str:
    """
    Determine severity based on the percentage
    of the dominant value.
    """
    if dominant_percentage < 95:
        return "low"
    elif dominant_percentage < 99:
        return "medium"
    else:
        return "high"


def check_constant_columns(df: pd.DataFrame) -> list:
    """
    Detect constant and near-constant columns.

    Constant:
        Exactly one unique non-null value.

    Near-constant:
        One value appears in at least 95% of
        non-null rows.

    Returns:
        A list of structured quality issues.
    """

    issues = []

    for column in df.columns:
        series = df[column]

        non_null = series.dropna()

        if non_null.empty:
            continue

        value_counts = non_null.value_counts()

        unique_count = int(len(value_counts))
        total_values = int(len(non_null))

        # ---------------------------------------------
        # Constant column
        # ---------------------------------------------
        if unique_count == 1:
            dominant_value = value_counts.index[0]
            dominant_count = int(value_counts.iloc[0])

            issues.append({
                "issue_type": "constant_column",
                "column": column,
                "count": dominant_count,
                "percentage": 100.0,
                "severity": "high",
                "evidence": {
                    "total_non_null_values": total_values,
                    "unique_value_count": unique_count,
                    "dominant_value": str(dominant_value),
                    "dominant_count": dominant_count,
                    "dominant_percentage": 100.0
                }
            })

            continue

        # ---------------------------------------------
        # Near-constant column
        # ---------------------------------------------
        dominant_value = value_counts.index[0]
        dominant_count = int(value_counts.iloc[0])

        dominant_percentage = (
            dominant_count / total_values
        ) * 100

        if dominant_percentage >= 95:
            severity = calculate_severity(
                dominant_percentage
            )

            issues.append({
                "issue_type": "near_constant_column",
                "column": column,
                "count": dominant_count,
                "percentage": round(
                    dominant_percentage,
                    2
                ),
                "severity": severity,
                "evidence": {
                    "total_non_null_values": total_values,
                    "unique_value_count": unique_count,
                    "dominant_value": str(dominant_value),
                    "dominant_count": dominant_count,
                    "dominant_percentage": round(
                        dominant_percentage,
                        2
                    )
                }
            })

    return issues