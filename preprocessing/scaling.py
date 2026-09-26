import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)


def scale_numeric_columns(
    df: pd.DataFrame,
    action: str,
    columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Scale numerical columns using an approved method.

    Supported actions:
    - standard
    - minmax
    """

    processed_df = df.copy()

    if columns is None:
        columns = list(
            processed_df.select_dtypes(
                include="number"
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

    if action not in {"standard", "minmax"}:
        raise ValueError(
            f"Unsupported scaling action: {action}"
        )

    for column in columns:

        if not pd.api.types.is_numeric_dtype(
            processed_df[column]
        ):
            raise ValueError(
                f"Column '{column}' must be numeric "
                "for scaling."
            )

    if action == "standard":
        scaler = StandardScaler()

    else:
        scaler = MinMaxScaler()

    processed_df[columns] = scaler.fit_transform(
        processed_df[columns]
    )

    changes = {
        "action": action,
        "affected_columns": columns,
        "rows_before": len(df),
        "rows_after": len(processed_df),
        "rows_removed": 0,
        "scaling_method": (
            "StandardScaler"
            if action == "standard"
            else "MinMaxScaler"
        )
    }

    return processed_df, changes