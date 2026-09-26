import pandas as pd


def calculate_severity(correlation_strength: float) -> str:
    """
    Determine severity based on absolute correlation strength.
    """
    if correlation_strength < 0.7:
        return "low"
    elif correlation_strength < 0.9:
        return "medium"
    else:
        return "high"


def classify_correlation(correlation: float) -> str:
    """
    Classify the strength and direction of a correlation.
    """
    absolute_correlation = abs(correlation)

    if absolute_correlation < 0.3:
        return "weak"
    elif absolute_correlation < 0.7:
        return "moderate"
    elif absolute_correlation < 0.9:
        return "strong"
    else:
        return "very_strong"


def check_correlations(
    df: pd.DataFrame,
    threshold: float = 0.7
) -> list:
    """
    Detect strong correlations between numerical columns.

    Args:
        df: Input DataFrame.
        threshold: Minimum absolute correlation to report.

    Returns:
        A list of structured correlation issues.
    """

    issues = []

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:
        return issues

    correlation_matrix = numeric_df.corr()

    columns = correlation_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            column_1 = columns[i]
            column_2 = columns[j]

            correlation = correlation_matrix.loc[
                column_1,
                column_2
            ]

            if pd.isna(correlation):
                continue

            absolute_correlation = abs(correlation)

            if absolute_correlation < threshold:
                continue

            correlation_strength = classify_correlation(
                correlation
            )

            severity = calculate_severity(
                absolute_correlation
            )

            issues.append({
                "issue_type": "strong_correlation",
                "column": column_1,
                "count": 2,
                "percentage": round(
                    absolute_correlation * 100,
                    2
                ),
                "severity": severity,
                "evidence": {
                    "column_1": column_1,
                    "column_2": column_2,
                    "correlation": round(
                        float(correlation),
                        4
                    ),
                    "absolute_correlation": round(
                        float(absolute_correlation),
                        4
                    ),
                    "correlation_strength": (
                        correlation_strength
                    ),
                    "threshold": threshold
                }
            })

    return issues