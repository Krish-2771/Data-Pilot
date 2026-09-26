import pandas as pd


def normalize_categorical_values(
    df: pd.DataFrame,
    columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Normalize categorical string values.

    String values are stripped of surrounding whitespace
    and converted to lowercase.
    """

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

    changes_by_column = {}

    for column in columns:

        column_dtype = processed_df[column].dtype

        is_categorical_column = (
            pd.api.types.is_object_dtype(column_dtype)
            or pd.api.types.is_string_dtype(column_dtype)
            or isinstance(
                column_dtype,
                pd.CategoricalDtype
            )
        )

        if not is_categorical_column:
            raise ValueError(
                f"Column '{column}' must be categorical "
                "or string type."
            )

        before = processed_df[column].copy()

        processed_df[column] = processed_df[column].map(
            lambda value: (
                value.strip().lower()
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
        "action": "normalize_categorical_values",
        "affected_columns": columns,
        "rows_before": len(df),
        "rows_after": len(processed_df),
        "rows_removed": 0,
        "changes_by_column": changes_by_column
    }

    return processed_df, changes