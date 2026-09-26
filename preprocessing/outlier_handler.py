import pandas as pd


def calculate_iqr_bounds(
    series: pd.Series,
    multiplier: float = 1.5
) -> tuple[float, float]:
    """
    Calculate IQR-based lower and upper bounds.
    """

    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(
            "Outlier handling requires a numeric column."
        )

    non_null = series.dropna()

    if non_null.empty:
        raise ValueError(
            "Cannot calculate outlier bounds for an empty column."
        )

    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return float(lower_bound), float(upper_bound)


def handle_outliers(
    df: pd.DataFrame,
    action: str,
    columns: list[str] | None = None,
    multiplier: float = 1.5
) -> tuple[pd.DataFrame, dict]:
    """
    Handle outliers using an approved preprocessing action.

    Supported actions:
    - remove_rows
    - clip

    Args:
        df: Input DataFrame.
        action: Approved outlier-handling action.
        columns: Numeric columns to process.
        multiplier: IQR multiplier.

    Returns:
        Processed DataFrame and change information.
    """

    processed_df = df.copy()

    rows_before = len(processed_df)

    if columns is None:
        columns = list(processed_df.select_dtypes(
            include="number"
        ).columns)

    invalid_columns = [
        column
        for column in columns
        if column not in processed_df.columns
    ]

    if invalid_columns:
        raise ValueError(
            f"Columns not found in dataset: {invalid_columns}"
        )

    if action not in {"remove_rows", "clip"}:
        raise ValueError(
            f"Unsupported outlier action: {action}"
        )

    bounds = {}
    outlier_counts_before = {}

    for column in columns:
        if not pd.api.types.is_numeric_dtype(
            processed_df[column]
        ):
            raise ValueError(
                f"Column '{column}' must be numeric for "
                "outlier handling."
            )

        lower_bound, upper_bound = calculate_iqr_bounds(
            processed_df[column],
            multiplier
        )

        bounds[column] = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

        outlier_mask = (
            (processed_df[column] < lower_bound)
            | (processed_df[column] > upper_bound)
        )

        outlier_counts_before[column] = int(
            outlier_mask.sum()
        )

    if action == "remove_rows":

        combined_outlier_mask = pd.Series(
            False,
            index=processed_df.index
        )

        for column in columns:
            lower_bound = bounds[column]["lower_bound"]
            upper_bound = bounds[column]["upper_bound"]

            column_outlier_mask = (
                (processed_df[column] < lower_bound)
                | (processed_df[column] > upper_bound)
            )

            combined_outlier_mask |= column_outlier_mask

        processed_df = processed_df.loc[
            ~combined_outlier_mask
        ].copy()

    elif action == "clip":

        for column in columns:
            lower_bound = bounds[column]["lower_bound"]
            upper_bound = bounds[column]["upper_bound"]

            processed_df[column] = processed_df[column].clip(
                lower=lower_bound,
                upper=upper_bound
            )

    rows_after = len(processed_df)

    outlier_counts_after = {}

    for column in columns:
        lower_bound = bounds[column]["lower_bound"]
        upper_bound = bounds[column]["upper_bound"]

        remaining_outliers = (
            (processed_df[column] < lower_bound)
            | (processed_df[column] > upper_bound)
        )

        outlier_counts_after[column] = int(
            remaining_outliers.sum()
        )

    changes = {
        "action": action,
        "affected_columns": columns,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "outlier_counts_before": outlier_counts_before,
        "outlier_counts_after": outlier_counts_after,
        "bounds": bounds
    }

    return processed_df, changes