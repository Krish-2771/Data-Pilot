import pandas as pd


def handle_duplicates(
    df: pd.DataFrame,
    action: str = "drop_duplicates",
    subset: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Handle duplicate rows using an approved preprocessing action.

    Supported action:
    - drop_duplicates

    Args:
        df: Input DataFrame.
        action: Approved duplicate-handling action.
        subset: Columns used to identify duplicates.
                If None, all columns are used.

    Returns:
        A tuple containing:
        - processed DataFrame
        - change information dictionary
    """

    processed_df = df.copy()

    rows_before = len(processed_df)

    # Handle empty subset same as None (use all columns)
    if not subset:
        subset = list(processed_df.columns)

    invalid_columns = [
        column
        for column in subset
        if column not in processed_df.columns
    ]

    if invalid_columns:
        raise ValueError(
            f"Columns not found in dataset: {invalid_columns}"
        )

    if action != "drop_duplicates":
        raise ValueError(
            f"Unsupported duplicate action: {action}"
        )

    duplicate_rows_before = int(
        processed_df.duplicated(
            subset=subset,
            keep="first"
        ).sum()
    )

    processed_df = processed_df.drop_duplicates(
        subset=subset,
        keep="first"
    )

    rows_after = len(processed_df)

    duplicate_rows_after = int(
        processed_df.duplicated(
            subset=subset,
            keep="first"
        ).sum()
    )

    changes = {
        "action": action,
        "affected_columns": subset,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "duplicate_rows_before": duplicate_rows_before,
        "duplicate_rows_after": duplicate_rows_after
    }

    return processed_df, changes