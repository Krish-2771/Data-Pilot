import pandas as pd
import numpy as np


def convert_to_numeric(
    df: pd.DataFrame,
    columns: list[str],
    errors: str = "raise"
) -> tuple[pd.DataFrame, dict]:
    """
    Convert columns to numeric type.

    Args:
        df: Input DataFrame.
        columns: Columns to convert.
        errors: "raise", "coerce", or "ignore".

    Returns:
        Processed DataFrame and change information.
    """
    processed_df = df.copy()

    invalid_columns = [
        column
        for column in columns
        if column not in processed_df.columns
    ]

    if invalid_columns:
        raise ValueError(
            f"Columns not found in dataset: {invalid_columns}"
        )

    changes_by_column = {}

    for column in columns:
        before = processed_df[column].copy()
        before_dtype = str(before.dtype)

        try:
            processed_df[column] = pd.to_numeric(
                processed_df[column],
                errors=errors
            )
        except Exception as e:
            raise ValueError(
                f"Failed to convert column '{column}' to numeric: {e}"
            )

        after_dtype = str(processed_df[column].dtype)

        changed = before_dtype != after_dtype
        changes_by_column[column] = {
            "dtype_before": before_dtype,
            "dtype_after": after_dtype,
            "changed": changed,
            "non_null_before": int(before.notna().sum()),
            "non_null_after": int(processed_df[column].notna().sum()),
        }

    changes = {
        "action": "convert_to_numeric",
        "affected_columns": columns,
        "rows_before": len(df),
        "rows_after": len(processed_df),
        "rows_removed": 0,
        "changes_by_column": changes_by_column
    }

    return processed_df, changes


def convert_to_datetime(
    df: pd.DataFrame,
    columns: list[str],
    format: str | None = None,
    errors: str = "raise"
) -> tuple[pd.DataFrame, dict]:
    """
    Convert columns to datetime type.

    Args:
        df: Input DataFrame.
        columns: Columns to convert.
        format: Datetime format string (None for mixed/inferred).
        errors: "raise", "coerce", or "ignore".

    Returns:
        Processed DataFrame and change information.
    """
    processed_df = df.copy()

    invalid_columns = [
        column
        for column in columns
        if column not in processed_df.columns
    ]

    if invalid_columns:
        raise ValueError(
            f"Columns not found in dataset: {invalid_columns}"
        )

    changes_by_column = {}

    for column in columns:
        before = processed_df[column].copy()
        before_dtype = str(before.dtype)

        try:
            processed_df[column] = pd.to_datetime(
                processed_df[column],
                format=format,
                errors=errors
            )
        except Exception as e:
            raise ValueError(
                f"Failed to convert column '{column}' to datetime: {e}"
            )

        after_dtype = str(processed_df[column].dtype)

        changed = before_dtype != after_dtype
        changes_by_column[column] = {
            "dtype_before": before_dtype,
            "dtype_after": after_dtype,
            "changed": changed,
            "non_null_before": int(before.notna().sum()),
            "non_null_after": int(processed_df[column].notna().sum()),
        }

    changes = {
        "action": "convert_to_datetime",
        "affected_columns": columns,
        "rows_before": len(df),
        "rows_after": len(processed_df),
        "rows_removed": 0,
        "changes_by_column": changes_by_column
    }

    return processed_df, changes


def convert_to_categorical(
    df: pd.DataFrame,
    columns: list[str]
) -> tuple[pd.DataFrame, dict]:
    """
    Convert columns to categorical type.

    Args:
        df: Input DataFrame.
        columns: Columns to convert.

    Returns:
        Processed DataFrame and change information.
    """
    processed_df = df.copy()

    invalid_columns = [
        column
        for column in columns
        if column not in processed_df.columns
    ]

    if invalid_columns:
        raise ValueError(
            f"Columns not found in dataset: {invalid_columns}"
        )

    changes_by_column = {}

    for column in columns:
        before = processed_df[column].copy()
        before_dtype = str(before.dtype)

        processed_df[column] = processed_df[column].astype("category")

        after_dtype = str(processed_df[column].dtype)

        changes_by_column[column] = {
            "dtype_before": before_dtype,
            "dtype_after": after_dtype,
            "changed": before_dtype != after_dtype,
            "categories": processed_df[column].cat.categories.tolist(),
        }

    changes = {
        "action": "convert_to_categorical",
        "affected_columns": columns,
        "rows_before": len(df),
        "rows_after": len(processed_df),
        "rows_removed": 0,
        "changes_by_column": changes_by_column
    }

    return processed_df, changes