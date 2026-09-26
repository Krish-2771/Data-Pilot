import pandas as pd


def detect_column_type(series: pd.Series) -> str:
    """
    Detect the logical type of a DataFrame column.

    Returns:
        numeric
        categorical
        text
        datetime
        boolean
        unknown
    """

    # Boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Remove missing values for further inspection
    non_null = series.dropna()

    if non_null.empty:
        return "unknown"

    # Try detecting datetime values
    converted_datetime = pd.to_datetime(
        non_null,
        errors="coerce",
        format="mixed"
    )

    datetime_ratio = converted_datetime.notna().mean()

    if datetime_ratio >= 0.8:
        return "datetime"

    # Convert values to strings for categorical/text analysis
    string_values = non_null.astype(str)

    unique_count = string_values.nunique()
    total_count = len(string_values)

    unique_ratio = unique_count / total_count

    # Categorical:
    # relatively small number of unique values
    if unique_count <= 20 or unique_ratio <= 0.05:
        return "categorical"

    # Otherwise treat as text
    return "text"


def profile_column(series: pd.Series) -> dict:
    """
    Generate a structured profile for a single column.

    The returned field names match ColumnSchema in
    schemas/dataset_schema.py.
    """

    column_type = detect_column_type(series)

    total_values = int(len(series))
    non_null_count = int(series.notna().sum())
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))

    missing_percentage = (
        (missing_count / total_values) * 100
        if total_values > 0
        else 0.0
    )

    return {
        "name": str(series.name),
        "data_type": str(series.dtype),
        "column_type": column_type,
        "non_null_count": non_null_count,
        "missing_count": missing_count,
        "missing_percentage": round(
            float(missing_percentage),
            2
        ),
        "unique_count": unique_count,
    }


def profile_columns(df: pd.DataFrame) -> list:
    """
    Generate profiles for all columns in a DataFrame.
    """

    profiles = []

    for column in df.columns:
        profiles.append(
            profile_column(df[column])
        )

    return profiles


def get_columns_by_type(
    df: pd.DataFrame,
    column_type: str
) -> list:
    """
    Return column names belonging to a particular logical type.
    """

    columns = []

    for column in df.columns:
        detected_type = detect_column_type(df[column])

        if detected_type == column_type:
            columns.append(column)

    return columns