import pandas as pd
import numpy as np


def trim_whitespace(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Remove leading and trailing whitespace from string columns.

    Args:
        df: Input DataFrame.
        columns: Columns to process. If None, all string columns.

    Returns:
        Processed DataFrame and change information.
    """
    processed_df = df.copy()

    if columns is None:
        columns = list(
            processed_df.select_dtypes(
                include=["object", "string"]
            ).columns
        )

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
        column_dtype = processed_df[column].dtype

        is_string_column = (
            pd.api.types.is_object_dtype(column_dtype)
            or pd.api.types.is_string_dtype(column_dtype)
        )

        if not is_string_column:
            raise ValueError(
                f"Column '{column}' must be string type "
                "for whitespace trimming."
            )

        before = processed_df[column].copy()

        processed_df[column] = processed_df[column].map(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                else value
            )
        )

        changed_count = int(
            (
                before.astype("string")
                != processed_df[column].astype("string")
            ).fillna(False).sum()
        )

        changes_by_column[column] = {
            "values_changed": changed_count
        }

    changes = {
        "action": "trim_whitespace",
        "affected_columns": columns,
        "rows_before": len(df),
        "rows_after": len(processed_df),
        "rows_removed": 0,
        "changes_by_column": changes_by_column
    }

    return processed_df, changes


def empty_strings_to_missing(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Convert empty strings and whitespace-only strings to NaN.

    Args:
        df: Input DataFrame.
        columns: Columns to process. If None, all string columns.

    Returns:
        Processed DataFrame and change information.
    """
    processed_df = df.copy()

    if columns is None:
        columns = list(
            processed_df.select_dtypes(
                include=["object", "string"]
            ).columns
        )

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
        column_dtype = processed_df[column].dtype

        is_string_column = (
            pd.api.types.is_object_dtype(column_dtype)
            or pd.api.types.is_string_dtype(column_dtype)
        )

        if not is_string_column:
            raise ValueError(
                f"Column '{column}' must be string type "
                "for empty string conversion."
            )

        before = processed_df[column].copy()

        # Convert empty strings and whitespace-only strings to NaN
        processed_df[column] = processed_df[column].map(
            lambda value: (
                np.nan
                if isinstance(value, str) and value.strip() == ""
                else value
            )
        )

        # Count changes: values that were non-null strings but became NaN
        changed_mask = (
            before.notna() &
            processed_df[column].isna() &
            before.astype(str).str.strip().eq("")
        )
        changed_count = int(changed_mask.sum())

        changes_by_column[column] = {
            "values_changed": changed_count
        }

    changes = {
        "action": "empty_strings_to_missing",
        "affected_columns": columns,
        "rows_before": len(df),
        "rows_after": len(processed_df),
        "rows_removed": 0,
        "changes_by_column": changes_by_column
    }

    return processed_df, changes


def normalize_categorical_case(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    case: str = "lower"
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize categorical string values to a consistent case.

    Args:
        df: Input DataFrame.
        columns: Columns to process. If None, all categorical columns.
        case: "lower", "upper", or "title" case.

    Returns:
        Processed DataFrame and change information.
    """
    if case not in {"lower", "upper", "title"}:
        raise ValueError(f"Unsupported case: {case}. Use 'lower', 'upper', or 'title'.")

    processed_df = df.copy()

    if columns is None:
        columns = list(
            processed_df.select_dtypes(
                include=["object", "string", "category"]
            ).columns
        )

    invalid_columns = [
        column
        for column in columns
        if column not in processed_df.columns
    ]

    if invalid_columns:
        raise ValueError(
            f"Columns not found in dataset: {invalid_columns}"
        )

    case_func = {
        "lower": str.lower,
        "upper": str.upper,
        "title": str.title,
    }[case]

    changes_by_column = {}

    for column in columns:
        column_dtype = processed_df[column].dtype

        is_categorical_column = (
            pd.api.types.is_object_dtype(column_dtype)
            or pd.api.types.is_string_dtype(column_dtype)
            or isinstance(column_dtype, pd.CategoricalDtype)
        )

        if not is_categorical_column:
            raise ValueError(
                f"Column '{column}' must be categorical "
                "or string type for case normalization."
            )

        before = processed_df[column].copy()

        processed_df[column] = processed_df[column].map(
            lambda value: (
                case_func(value.strip())
                if isinstance(value, str)
                else value
            )
        )

        changed_count = int(
            (
                before.astype("string")
                != processed_df[column].astype("string")
            ).fillna(False).sum()
        )

        changes_by_column[column] = {
            "values_changed": changed_count
        }

    changes = {
        "action": f"normalize_categorical_case_{case}",
        "affected_columns": columns,
        "rows_before": len(df),
        "rows_after": len(processed_df),
        "rows_removed": 0,
        "changes_by_column": changes_by_column
    }

    return processed_df, changes