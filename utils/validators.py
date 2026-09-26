import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that the input is a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")


def validate_columns(
    df: pd.DataFrame,
    columns: list[str] | None,
) -> None:
    """
    Validate that requested columns exist in the DataFrame.
    """
    validate_dataframe(df)

    if columns is None:
        return

    invalid_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if invalid_columns:
        raise ValueError(
            f"Columns not found in dataset: {invalid_columns}"
        )


def validate_numeric_columns(
    df: pd.DataFrame,
    columns: list[str] | None,
) -> None:
    """
    Validate that requested columns exist and are numeric.
    """
    validate_columns(df, columns)

    if columns is None:
        return

    non_numeric_columns = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise ValueError(
            f"Columns must be numeric: {non_numeric_columns}"
        )


def validate_categorical_columns(
    df: pd.DataFrame,
    columns: list[str] | None,
) -> None:
    """
    Validate that requested columns exist and are categorical/string columns.
    """
    validate_columns(df, columns)

    if columns is None:
        return

    invalid_columns = [
        column
        for column in columns
        if not (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
            or isinstance(df[column].dtype, pd.CategoricalDtype)
        )
    ]

    if invalid_columns:
        raise ValueError(
            f"Columns must be categorical or string type: "
            f"{invalid_columns}"
        )