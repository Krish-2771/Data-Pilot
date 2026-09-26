import pandas as pd


def calculate_severity(cardinality_percentage: float) -> str:
    """
    Determine severity based on the percentage
    of unique values.
    """
    if cardinality_percentage <= 20:
        return "low"
    elif cardinality_percentage <= 50:
        return "medium"
    elif cardinality_percentage <= 80:
        return "high"
    else:
        return "critical"


def check_cardinality(df: pd.DataFrame) -> list:
    """
    Detect unusually high or low cardinality in columns.

    High cardinality:
        More than 80% of non-null values are unique.

    Low cardinality:
        Categorical/text column with 2 or fewer unique values.

    Returns:
        A list of structured quality issues.
    """

    issues = []

    for column in df.columns:
        series = df[column]

        non_null = series.dropna()

        if non_null.empty:
            continue

        total_values = len(non_null)
        unique_count = int(non_null.nunique())
        cardinality_percentage = (
            unique_count / total_values
        ) * 100

        # ---------------------------------------------
        # High cardinality
        # ---------------------------------------------
        if cardinality_percentage > 80:
            severity = calculate_severity(
                cardinality_percentage
            )

            issues.append({
                "issue_type": "high_cardinality",
                "column": column,
                "count": unique_count,
                "percentage": round(
                    cardinality_percentage,
                    2
                ),
                "severity": severity,
                "evidence": {
                    "total_non_null_values": int(
                        total_values
                    ),
                    "unique_value_count": unique_count,
                    "cardinality_percentage": round(
                        cardinality_percentage,
                        2
                    )
                }
            })

            continue

        # ---------------------------------------------
        # Low cardinality
        # ---------------------------------------------
        if (
            not pd.api.types.is_numeric_dtype(series)
            and unique_count <= 2
        ):
            issues.append({
                "issue_type": "low_cardinality",
                "column": column,
                "count": unique_count,
                "percentage": round(
                    cardinality_percentage,
                    2
                ),
                "severity": "low",
                "evidence": {
                    "total_non_null_values": int(
                        total_values
                    ),
                    "unique_value_count": unique_count,
                    "cardinality_percentage": round(
                        cardinality_percentage,
                        2
                    )
                }
            })

    return issues