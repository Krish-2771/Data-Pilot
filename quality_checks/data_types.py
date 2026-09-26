import pandas as pd


def check_data_types(df: pd.DataFrame) -> list:
    """
    Detect potential data type problems in a DataFrame.

    Checks for:
    - Numeric-looking columns stored as text
    - Date-looking columns stored as text
    - Mixed data types inside object columns

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
        # Already numeric
        # -------------------------------------------------

        if pd.api.types.is_numeric_dtype(series):
            continue

        # Convert values to strings for analysis
        string_values = non_null.astype(str).str.strip()

        # -------------------------------------------------
        # Check whether values are numeric-looking
        # -------------------------------------------------

        numeric_conversion = pd.to_numeric(
            string_values,
            errors="coerce"
        )

        numeric_count = int(
            numeric_conversion.notna().sum()
        )

        numeric_percentage = (
            numeric_count / len(string_values)
        ) * 100

        if numeric_percentage >= 80:

            issues.append({
                "issue_type": "possible_numeric_type_mismatch",
                "column": column,
                "count": numeric_count,
                "percentage": round(
                    numeric_percentage,
                    2
                ),
                "severity": "medium",
                "evidence": {
                    "actual_dtype": str(series.dtype),
                    "numeric_values": numeric_count,
                    "total_non_null_values": len(string_values),
                    "numeric_value_percentage": round(
                        numeric_percentage,
                        2
                    )
                }
            })

            continue

        # -------------------------------------------------
        # Check whether values are datetime-looking
        # -------------------------------------------------

        datetime_conversion = pd.to_datetime(
            string_values,
            errors="coerce",
            format="mixed"
        )

        datetime_count = int(
            datetime_conversion.notna().sum()
        )

        datetime_percentage = (
            datetime_count / len(string_values)
        ) * 100

        if datetime_percentage >= 80:

            issues.append({
                "issue_type": "possible_datetime_type_mismatch",
                "column": column,
                "count": datetime_count,
                "percentage": round(
                    datetime_percentage,
                    2
                ),
                "severity": "medium",
                "evidence": {
                    "actual_dtype": str(series.dtype),
                    "datetime_values": datetime_count,
                    "total_non_null_values": len(string_values),
                    "datetime_value_percentage": round(
                        datetime_percentage,
                        2
                    )
                }
            })

            continue

        # -------------------------------------------------
        # Check for mixed data types
        # -------------------------------------------------

        value_types = (
            non_null
            .map(lambda value: type(value).__name__)
            .nunique()
        )

        if value_types > 1:

            issues.append({
                "issue_type": "mixed_data_types",
                "column": column,
                "count": int(len(non_null)),
                "percentage": 100.0,
                "severity": "medium",
                "evidence": {
                    "actual_dtype": str(series.dtype),
                    "different_python_types": value_types
                }
            })

    return issues