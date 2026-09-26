import pandas as pd


def is_categorical_column(series: pd.Series) -> bool:
    """
    Check whether a Series can be treated as categorical data.
    Supports object, string, and category dtypes.
    """

    dtype = series.dtype

    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def label_encode(
    df: pd.DataFrame,
    columns: list[str]
) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns using deterministic integer mappings.
    """

    processed_df = df.copy()
    mappings = {}

    for column in columns:

        if column not in processed_df.columns:
            raise ValueError(
                f"Column '{column}' not found in dataset."
            )

        if not is_categorical_column(processed_df[column]):
            raise ValueError(
                f"Column '{column}' must be categorical "
                "or string type."
            )

        unique_values = sorted(
            processed_df[column]
            .dropna()
            .unique()
            .tolist(),
            key=lambda value: str(value)
        )

        mapping = {
            value: index
            for index, value in enumerate(unique_values)
        }

        processed_df[column] = processed_df[column].map(
            mapping
        )

        mappings[column] = {
            str(key): value
            for key, value in mapping.items()
        }

    changes = {
        "action": "label_encoding",
        "affected_columns": columns,
        "rows_before": len(df),
        "rows_after": len(processed_df),
        "rows_removed": 0,
        "mappings": mappings
    }

    return processed_df, changes


def one_hot_encode(
    df: pd.DataFrame,
    columns: list[str]
) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical columns using one-hot encoding.
    """

    processed_df = df.copy()

    for column in columns:

        if column not in processed_df.columns:
            raise ValueError(
                f"Column '{column}' not found in dataset."
            )

        if not is_categorical_column(processed_df[column]):
            raise ValueError(
                f"Column '{column}' must be categorical "
                "or string type."
            )

    processed_df = pd.get_dummies(
        processed_df,
        columns=columns,
        dtype=int
    )

    changes = {
        "action": "one_hot_encoding",
        "affected_columns": columns,
        "rows_before": len(df),
        "rows_after": len(processed_df),
        "rows_removed": 0,
        "new_columns": [
            column
            for column in processed_df.columns
            if column not in df.columns
        ]
    }

    return processed_df, changes


def encode_categorical(
    df: pd.DataFrame,
    action: str,
    columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Apply an approved categorical encoding action.

    Supported actions:
    - label_encoding
    - one_hot_encoding
    """

    if columns is None:
        columns = list(
            df.select_dtypes(
                include=["object", "string", "category"]
            ).columns
        )

    if action == "label_encoding":
        return label_encode(df, columns)

    if action == "one_hot_encoding":
        return one_hot_encode(df, columns)

    raise ValueError(
        f"Unsupported encoding action: {action}"
    )