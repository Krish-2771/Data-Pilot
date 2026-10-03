import pandas as pd
import numpy as np


def handle_missing_values(
    df: pd.DataFrame,
    action: str,
    columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Handle missing values using an approved preprocessing action.

    Supported actions:
    - drop_rows
    - fill_mean
    - fill_median
    - fill_mode
    - fill_unknown (categorical - adds "Unknown" category)

    Args:
        df: Input DataFrame.
        action: Approved missing-value action.
        columns: Columns on which to apply the action.
                 If None, all applicable columns are used.

    Returns:
        A tuple containing:
        - processed DataFrame
        - change information dictionary
    """

    processed_df = df.copy()

    rows_before = len(processed_df)

    if columns is None:
        columns = list(processed_df.columns)

    invalid_columns = [
        column
        for column in columns
        if column not in processed_df.columns
    ]

    if invalid_columns:
        raise ValueError(
            f"Columns not found in dataset: {invalid_columns}"
        )

    if action == "drop_rows":
        missing_before = int(
            processed_df[columns].isna().sum().sum()
        )

        processed_df = processed_df.dropna(
            subset=columns
        )

        rows_after = len(processed_df)

        changes = {
            "action": action,
            "affected_columns": columns,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
            "missing_values_before": missing_before,
            "missing_values_after": int(
                processed_df[columns].isna().sum().sum()
            )
        }

        return processed_df, changes

    for column in columns:
        if not processed_df[column].isna().any():
            continue

        if action == "fill_mean":
            if not pd.api.types.is_numeric_dtype(
                processed_df[column]
            ):
                raise ValueError(
                    f"Column '{column}' must be numeric "
                    "for fill_mean."
                )

            fill_value = processed_df[column].mean()

        elif action == "fill_median":
            if not pd.api.types.is_numeric_dtype(
                processed_df[column]
            ):
                raise ValueError(
                    f"Column '{column}' must be numeric "
                    "for fill_median."
                )

            fill_value = processed_df[column].median()

        elif action == "fill_mode":
            mode_values = processed_df[column].mode()

            if mode_values.empty:
                raise ValueError(
                    f"Cannot determine mode for "
                    f"column '{column}'."
                )

            fill_value = mode_values.iloc[0]

        elif action == "fill_unknown":
            # For categorical columns, add "Unknown" as a new category
            fill_value = "Unknown"

            # If column is Categorical, add "Unknown" to categories first
            if isinstance(processed_df[column].dtype, pd.CategoricalDtype):
                processed_df[column] = processed_df[column].cat.add_categories([fill_value])

        else:
            raise ValueError(
                f"Unsupported missing-value action: {action}"
            )

        processed_df[column] = (
            processed_df[column].fillna(fill_value)
        )

    if action not in {
        "fill_mean",
        "fill_median",
        "fill_mode",
        "fill_unknown"
    }:
        raise ValueError(
            f"Unsupported missing-value action: {action}"
        )

    rows_after = len(processed_df)

    changes = {
        "action": action,
        "affected_columns": columns,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "missing_values_before": int(
            df[columns].isna().sum().sum()
        ),
        "missing_values_after": int(
            processed_df[columns].isna().sum().sum()
        )
    }

    return processed_df, changes