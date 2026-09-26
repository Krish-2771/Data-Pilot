import pandas as pd
import re


ID_NAME_PATTERNS = [
    r"(^|_)id($|_)",
    r"identifier",
    r"(^|_)code($|_)"
]


def calculate_severity(
    uniqueness_percentage: float
) -> str:
    """
    Determine severity based on uniqueness percentage.
    """
    if uniqueness_percentage <= 80:
        return "low"
    elif uniqueness_percentage <= 95:
        return "medium"
    else:
        return "high"


def has_id_like_name(column: str) -> bool:
    """
    Check whether a column name looks like an identifier.
    """
    column_name = str(column).lower().strip()

    for pattern in ID_NAME_PATTERNS:
        if re.search(pattern, column_name):
            return True

    return False


def is_sequential_numeric(series: pd.Series) -> bool:
    """
    Check whether a numeric series contains sequential values.
    """
    values = series.dropna()

    if len(values) < 2:
        return False

    if not pd.api.types.is_numeric_dtype(values):
        return False

    sorted_values = values.sort_values()

    differences = sorted_values.diff().dropna()

    return bool((differences == 1).all())


def check_id_detection(df: pd.DataFrame) -> list:
    """
    Detect columns that are likely to be identifiers.

    Signals considered:
    - ID-like column name
    - Very high uniqueness
    - Sequential numeric values

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

        uniqueness_percentage = (
            unique_count / total_values
        ) * 100

        name_signal = has_id_like_name(column)

        sequential_signal = is_sequential_numeric(
            series
        )

        high_uniqueness_signal = (
            uniqueness_percentage >= 95
        )

        signal_count = sum([
            name_signal,
            sequential_signal,
            high_uniqueness_signal
        ])

        # Require at least two signals
        # to reduce false positives.
        if signal_count < 2:
            continue

        severity = calculate_severity(
            uniqueness_percentage
        )

        issues.append({
            "issue_type": "possible_identifier",
            "column": column,
            "count": unique_count,
            "percentage": round(
                float(uniqueness_percentage),
                2
            ),
            "severity": severity,
            "evidence": {
                "total_non_null_values": int(
                    total_values
                ),
                "unique_value_count": unique_count,
                "uniqueness_percentage": round(
                    float(uniqueness_percentage),
                    2
                ),
                "signals": {
                    "id_like_name": name_signal,
                    "sequential_numeric": sequential_signal,
                    "high_uniqueness": (
                        high_uniqueness_signal
                    )
                }
            }
        })

    return issues