import pandas as pd


def calculate_severity(inconsistent_percentage: float) -> str:
    """
    Determine severity based on the percentage
    of inconsistent categorical values.
    """
    if inconsistent_percentage == 0:
        return "none"
    elif inconsistent_percentage <= 5:
        return "low"
    elif inconsistent_percentage <= 20:
        return "medium"
    elif inconsistent_percentage <= 50:
        return "high"
    else:
        return "critical"


def check_categorical_consistency(
    df: pd.DataFrame
) -> list:
    """
    Detect inconsistencies in categorical columns.

    Values that differ only because of capitalization
    or surrounding whitespace are treated as inconsistent.

    Returns:
        A list of structured quality issues.
    """

    issues = []

    for column in df.columns:
        series = df[column]

        # Skip numeric columns
        if pd.api.types.is_numeric_dtype(series):
            continue

        # Remove missing values
        non_null = series.dropna()

        if non_null.empty:
            continue

        # Convert values to strings and remove whitespace
        values = non_null.astype(str).str.strip()

        # Normalized values for comparison
        normalized_values = values.str.lower()

        # Group original values by normalized value
        groups = {}

        for original, normalized in zip(
            values,
            normalized_values
        ):
            if normalized not in groups:
                groups[normalized] = set()

            groups[normalized].add(original)

        # Find groups containing multiple representations
        inconsistent_groups = {
            normalized: originals
            for normalized, originals in groups.items()
            if len(originals) > 1
        }

        if not inconsistent_groups:
            continue

        # Count the values belonging to inconsistent groups
        inconsistent_count = sum(
            1
            for normalized in normalized_values
            if normalized in inconsistent_groups
        )

        inconsistent_percentage = (
            inconsistent_count / len(values)
        ) * 100

        severity = calculate_severity(
            inconsistent_percentage
        )

        issues.append({
            "issue_type": "categorical_inconsistency",
            "column": column,
            "count": int(inconsistent_count),
            "percentage": round(
                inconsistent_percentage,
                2
            ),
            "severity": severity,
            "evidence": {
                "total_non_null_values": int(
                    len(values)
                ),
                "inconsistent_value_count": int(
                    inconsistent_count
                ),
                "inconsistent_percentage": round(
                    inconsistent_percentage,
                    2
                ),
                "inconsistent_groups": {
                    normalized: sorted(
                        list(originals)
                    )
                    for normalized, originals
                    in inconsistent_groups.items()
                }
            }
        })

    return issues