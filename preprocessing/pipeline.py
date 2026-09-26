import pandas as pd

from preprocessing.missing_handler import (
    handle_missing_values
)

from preprocessing.duplicate_handler import (
    handle_duplicates
)

from preprocessing.outlier_handler import (
    handle_outliers
)

from preprocessing.categorical_handler import (
    normalize_categorical_values
)

from preprocessing.encoding import (
    encode_categorical
)

from preprocessing.scaling import (
    scale_numeric_columns
)


def apply_preprocessing_action(
    df: pd.DataFrame,
    action: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Apply one approved preprocessing action.

    Expected action format:

    {
        "action": "fill_mean",
        "columns": ["Age"]
    }

    Additional parameters can be provided depending
    on the preprocessing operation.
    """

    if not isinstance(action, dict):
        raise ValueError(
            "Each preprocessing action must be a dictionary."
        )

    action_name = action.get("action")

    if not action_name:
        raise ValueError(
            "Preprocessing action must contain 'action'."
        )

    columns = action.get("columns")

    if action_name in {
        "fill_mean",
        "fill_median",
        "fill_mode",
        "drop_rows"
    }:
        return handle_missing_values(
            df,
            action=action_name,
            columns=columns
        )

    if action_name == "drop_duplicates":
        return handle_duplicates(
            df,
            action=action_name,
            subset=columns
        )

    if action_name in {
        "remove_outliers",
        "clip_outliers"
    }:
        actual_action = (
            "remove_rows"
            if action_name == "remove_outliers"
            else "clip"
        )

        return handle_outliers(
            df,
            action=actual_action,
            columns=columns,
            multiplier=action.get(
                "multiplier",
                1.5
            )
        )

    if action_name == "normalize_categorical_values":
        return normalize_categorical_values(
            df,
            columns=columns
        )

    if action_name in {
        "label_encoding",
        "one_hot_encoding"
    }:
        return encode_categorical(
            df,
            action=action_name,
            columns=columns
        )

    if action_name in {
        "standard",
        "minmax"
    }:
        return scale_numeric_columns(
            df,
            action=action_name,
            columns=columns
        )

    raise ValueError(
        f"Unsupported preprocessing action: {action_name}"
    )


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    actions: list[dict]
) -> tuple[pd.DataFrame, dict]:
    """
    Execute a sequence of approved preprocessing actions.

    The original DataFrame is never modified.
    """

    if not isinstance(actions, list):
        raise ValueError(
            "Actions must be provided as a list."
        )

    processed_df = df.copy()

    rows_before = len(processed_df)

    action_logs = []

    for action in actions:

        processed_df, changes = (
            apply_preprocessing_action(
                processed_df,
                action
            )
        )

        action_logs.append(changes)

    rows_after = len(processed_df)

    summary = {
        "actions_requested": len(actions),
        "actions_applied": len(action_logs),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "action_logs": action_logs
    }

    return processed_df, summary