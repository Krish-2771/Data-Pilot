import pandas as pd
import pytest

from preprocessing.missing_handler import handle_missing_values
from preprocessing.duplicate_handler import handle_duplicates
from preprocessing.outlier_handler import handle_outliers
from preprocessing.categorical_handler import normalize_categorical_values
from preprocessing.encoding import (
    encode_categorical,
    label_encode,
    one_hot_encode,
)
from preprocessing.scaling import scale_numeric_columns
from preprocessing.pipeline import (
    apply_preprocessing_action,
    run_preprocessing_pipeline,
)


def test_fill_mean():
    df = pd.DataFrame({
        "Age": [20, 30, None, 40],
    })

    processed_df, changes = handle_missing_values(
        df,
        action="fill_mean",
        columns=["Age"],
    )

    assert processed_df["Age"].isna().sum() == 0
    assert processed_df["Age"].iloc[2] == 30
    assert changes["missing_values_before"] == 1
    assert changes["missing_values_after"] == 0


def test_fill_median():
    df = pd.DataFrame({
        "Age": [10, 20, None, 100],
    })

    processed_df, changes = handle_missing_values(
        df,
        action="fill_median",
        columns=["Age"],
    )

    assert processed_df["Age"].isna().sum() == 0
    assert processed_df["Age"].iloc[2] == 20
    assert changes["missing_values_after"] == 0


def test_fill_mode():
    df = pd.DataFrame({
        "Department": ["IT", "HR", "IT", None],
    })

    processed_df, changes = handle_missing_values(
        df,
        action="fill_mode",
        columns=["Department"],
    )

    assert processed_df["Department"].isna().sum() == 0
    assert processed_df["Department"].iloc[3] == "IT"


def test_drop_missing_rows():
    df = pd.DataFrame({
        "Age": [20, None, 30],
        "Name": ["A", "B", "C"],
    })

    processed_df, changes = handle_missing_values(
        df,
        action="drop_rows",
        columns=["Age"],
    )

    assert len(processed_df) == 2
    assert changes["rows_removed"] == 1
    assert processed_df["Age"].isna().sum() == 0


def test_drop_duplicates():
    df = pd.DataFrame({
        "Name": ["A", "B", "B"],
        "Age": [20, 30, 30],
    })

    processed_df, changes = handle_duplicates(df)

    assert len(processed_df) == 2
    assert changes["duplicate_rows_before"] == 1
    assert changes["duplicate_rows_after"] == 0
    assert changes["rows_removed"] == 1


def test_outlier_clipping():
    df = pd.DataFrame({
        "Age": [20, 21, 22, 23, 150],
    })

    processed_df, changes = handle_outliers(
        df,
        action="clip",
        columns=["Age"],
    )

    assert processed_df["Age"].max() < 150
    assert changes["rows_before"] == 5
    assert changes["rows_after"] == 5
    assert changes["outlier_counts_before"]["Age"] == 1
    assert changes["outlier_counts_after"]["Age"] == 0


def test_outlier_remove_rows():
    df = pd.DataFrame({
        "Age": [20, 21, 22, 23, 150],
    })

    processed_df, changes = handle_outliers(
        df,
        action="remove_rows",
        columns=["Age"],
    )

    assert len(processed_df) == 4
    assert changes["rows_before"] == 5
    assert changes["rows_after"] == 4
    assert changes["rows_removed"] == 1


def test_categorical_normalization():
    df = pd.DataFrame({
        "Department": ["IT", "it", "HR", "hr"],
    })

    processed_df, changes = normalize_categorical_values(
        df,
        columns=["Department"],
    )

    assert list(processed_df["Department"]) == [
        "it",
        "it",
        "hr",
        "hr",
    ]

    assert changes["affected_columns"] == ["Department"]

    # Only "IT" and "HR" changed.
    "it" == processed_df["Department"].iloc[0]
    assert changes["changes_by_column"]["Department"]["values_changed"] == 2


def test_label_encoding():
    df = pd.DataFrame({
        "Department": ["IT", "HR", "IT"],
    })

    processed_df, changes = label_encode(
        df,
        columns=["Department"],
    )

    assert pd.api.types.is_numeric_dtype(
        processed_df["Department"]
    )

    assert set(processed_df["Department"]) == {0, 1}
    assert changes["action"] == "label_encoding"
    assert changes["affected_columns"] == ["Department"]
    assert "Department" in changes["mappings"]


def test_one_hot_encoding():
    df = pd.DataFrame({
        "Department": ["IT", "HR", "IT"],
    })

    processed_df, changes = one_hot_encode(
        df,
        columns=["Department"],
    )

    assert "Department_IT" in processed_df.columns
    assert "Department_HR" in processed_df.columns
    assert "Department" not in processed_df.columns

    assert changes["action"] == "one_hot_encoding"
    assert changes["affected_columns"] == ["Department"]


def test_encode_categorical_action():
    df = pd.DataFrame({
        "Department": ["IT", "HR", "IT"],
    })

    processed_df, changes = encode_categorical(
        df,
        action="one_hot_encoding",
        columns=["Department"],
    )

    assert "Department_IT" in processed_df.columns
    assert "Department_HR" in processed_df.columns
    assert changes["action"] == "one_hot_encoding"


def test_minmax_scaling():
    df = pd.DataFrame({
        "Age": [10, 20, 30, 40],
    })

    processed_df, changes = scale_numeric_columns(
        df,
        action="minmax",
        columns=["Age"],
    )

    assert processed_df["Age"].min() == pytest.approx(0.0)
    assert processed_df["Age"].max() == pytest.approx(1.0)
    assert changes["affected_columns"] == ["Age"]
    assert changes["scaling_method"] == "MinMaxScaler"


def test_standard_scaling():
    df = pd.DataFrame({
        "Age": [10, 20, 30, 40],
    })

    processed_df, changes = scale_numeric_columns(
        df,
        action="standard",
        columns=["Age"],
    )

    assert processed_df["Age"].mean() == pytest.approx(0.0)
    assert changes["affected_columns"] == ["Age"]
    assert changes["scaling_method"] == "StandardScaler"


def test_pipeline_missing_value_action():
    df = pd.DataFrame({
        "Age": [20, None, 40],
    })

    processed_df, changes = apply_preprocessing_action(
        df,
        {
            "action": "fill_mean",
            "columns": ["Age"],
        },
    )

    assert processed_df["Age"].isna().sum() == 0
    assert changes["action"] == "fill_mean"


def test_pipeline_duplicate_action():
    df = pd.DataFrame({
        "Name": ["A", "B", "B"],
        "Age": [20, 30, 30],
    })

    processed_df, changes = apply_preprocessing_action(
        df,
        {
            "action": "drop_duplicates",
        },
    )

    assert len(processed_df) == 2
    assert changes["rows_removed"] == 1


def test_pipeline_encoding_action():
    df = pd.DataFrame({
        "Department": ["IT", "HR", "IT"],
    })

    processed_df, changes = apply_preprocessing_action(
        df,
        {
            "action": "one_hot_encoding",
            "columns": ["Department"],
        },
    )

    assert "Department_IT" in processed_df.columns
    assert "Department_HR" in processed_df.columns
    assert changes["action"] == "one_hot_encoding"


def test_pipeline_scaling_action():
    df = pd.DataFrame({
        "Age": [10, 20, 30, 40],
    })

    processed_df, changes = apply_preprocessing_action(
        df,
        {
            "action": "minmax",
            "columns": ["Age"],
        },
    )

    assert processed_df["Age"].min() == pytest.approx(0.0)
    assert processed_df["Age"].max() == pytest.approx(1.0)


def test_run_preprocessing_pipeline():
    df = pd.DataFrame({
        "Age": [20, None, 40],
        "Department": ["IT", "HR", "IT"],
    })

    original = df.copy()

    processed_df, summary = run_preprocessing_pipeline(
        df,
        [
            {
                "action": "fill_mean",
                "columns": ["Age"],
            },
            {
                "action": "one_hot_encoding",
                "columns": ["Department"],
            },
        ],
    )

    assert processed_df["Age"].isna().sum() == 0
    assert "Department_IT" in processed_df.columns
    assert "Department_HR" in processed_df.columns

    assert summary["actions_requested"] == 2
    assert summary["actions_applied"] == 2
    assert summary["rows_before"] == 3
    assert summary["rows_after"] == 3

    # Pipeline must not modify the original DataFrame.
    pd.testing.assert_frame_equal(df, original)


def test_original_dataframe_is_not_modified():
    df = pd.DataFrame({
        "Age": [20, None, 40],
    })

    original = df.copy()

    handle_missing_values(
        df,
        action="fill_mean",
        columns=["Age"],
    )

    pd.testing.assert_frame_equal(df, original)


def test_invalid_missing_action():
    df = pd.DataFrame({
        "Age": [20, 30, 40],
    })

    with pytest.raises(ValueError):
        handle_missing_values(
            df,
            action="invalid_action",
            columns=["Age"],
        )


def test_invalid_pipeline_action():
    df = pd.DataFrame({
        "Age": [20, 30, 40],
    })

    with pytest.raises(ValueError):
        apply_preprocessing_action(
            df,
            {
                "action": "invalid_action",
                "columns": ["Age"],
            },
        )