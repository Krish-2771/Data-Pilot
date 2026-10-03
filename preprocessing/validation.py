import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class ValidationStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    status: ValidationStatus
    message: str
    details: dict = None

    def to_dict(self) -> dict:
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details or {}
        }


@dataclass
class ValidationReport:
    """Complete validation report."""
    results: list[ValidationResult]

    @property
    def overall_status(self) -> ValidationStatus:
        """Determine overall status from individual results."""
        statuses = [r.status for r in self.results]
        if ValidationStatus.FAIL in statuses:
            return ValidationStatus.FAIL
        if ValidationStatus.WARNING in statuses:
            return ValidationStatus.WARNING
        return ValidationStatus.PASS

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status.value,
            "results": [r.to_dict() for r in self.results]
        }


def validate_columns_exist(
    df: pd.DataFrame,
    expected_columns: list[str]
) -> ValidationResult:
    """Validate that expected columns exist in the DataFrame."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        return ValidationResult(
            check_name="columns_exist",
            status=ValidationStatus.FAIL,
            message=f"Missing expected columns: {missing}",
            details={"missing_columns": missing, "current_columns": list(df.columns)}
        )
    return ValidationResult(
        check_name="columns_exist",
        status=ValidationStatus.PASS,
        message="All expected columns present",
        details={"columns": list(df.columns)}
    )


def validate_no_missing_values(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    threshold: float = 0.0
) -> ValidationResult:
    """Validate that specified columns have no (or minimal) missing values."""
    if columns is None:
        columns = list(df.columns)

    issues = []
    for column in columns:
        if column not in df.columns:
            continue
        missing_pct = df[column].isna().mean() * 100
        if missing_pct > threshold:
            issues.append({
                "column": column,
                "missing_percentage": round(missing_pct, 2)
            })

    if issues:
        return ValidationResult(
            check_name="no_missing_values",
            status=ValidationStatus.FAIL if any(i["missing_percentage"] > 5 for i in issues) else ValidationStatus.WARNING,
            message=f"Missing values found in {len(issues)} column(s)",
            details={"issues": issues}
        )
    return ValidationResult(
        check_name="no_missing_values",
        status=ValidationStatus.PASS,
        message="No missing values above threshold",
        details={"threshold_percentage": threshold}
    )


def validate_no_duplicates(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None
) -> ValidationResult:
    """Validate that DataFrame has no duplicate rows."""
    duplicate_count = int(df.duplicated(subset=subset, keep="first").sum())
    if duplicate_count > 0:
        return ValidationResult(
            check_name="no_duplicates",
            status=ValidationStatus.FAIL,
            message=f"Found {duplicate_count} duplicate row(s)",
            details={"duplicate_count": duplicate_count, "total_rows": len(df)}
        )
    return ValidationResult(
        check_name="no_duplicates",
        status=ValidationStatus.PASS,
        message="No duplicate rows found",
        details={"total_rows": len(df)}
    )


def validate_data_types(
    df: pd.DataFrame,
    expected_types: dict[str, str]
) -> ValidationResult:
    """Validate that columns have expected data types."""
    issues = []
    for column, expected_type in expected_types.items():
        if column not in df.columns:
            issues.append({
                "column": column,
                "issue": "column_missing",
                "expected_type": expected_type
            })
            continue

        actual_type = str(df[column].dtype)
        type_match = False

        if expected_type == "numeric":
            type_match = pd.api.types.is_numeric_dtype(df[column])
        elif expected_type == "datetime":
            type_match = pd.api.types.is_datetime64_any_dtype(df[column])
        elif expected_type == "categorical":
            type_match = pd.api.types.is_categorical_dtype(df[column])
        elif expected_type == "string":
            type_match = pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column])
        else:
            type_match = actual_type == expected_type

        if not type_match:
            issues.append({
                "column": column,
                "expected_type": expected_type,
                "actual_type": actual_type
            })

    if issues:
        return ValidationResult(
            check_name="data_types",
            status=ValidationStatus.FAIL,
            message=f"Data type mismatch in {len(issues)} column(s)",
            details={"issues": issues}
        )
    return ValidationResult(
        check_name="data_types",
        status=ValidationStatus.PASS,
        message="All columns have expected data types",
        details={}
    )


def validate_numeric_range(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> ValidationResult:
    """Validate that numeric column values are within expected range."""
    if column not in df.columns:
        return ValidationResult(
            check_name="numeric_range",
            status=ValidationStatus.FAIL,
            message=f"Column '{column}' not found",
            details={"column": column}
        )

    if not pd.api.types.is_numeric_dtype(df[column]):
        return ValidationResult(
            check_name="numeric_range",
            status=ValidationStatus.FAIL,
            message=f"Column '{column}' is not numeric",
            details={"column": column, "actual_type": str(df[column].dtype)}
        )

    series = df[column].dropna()
    if series.empty:
        return ValidationResult(
            check_name="numeric_range",
            status=ValidationStatus.WARNING,
            message=f"Column '{column}' has no non-null values",
            details={"column": column}
        )

    issues = []
    if min_value is not None:
        below = series[series < min_value]
        if len(below) > 0:
            issues.append({
                "issue": "below_minimum",
                "count": len(below),
                "min_expected": min_value,
                "min_actual": float(series.min())
            })

    if max_value is not None:
        above = series[series > max_value]
        if len(above) > 0:
            issues.append({
                "issue": "above_maximum",
                "count": len(above),
                "max_expected": max_value,
                "max_actual": float(series.max())
            })

    if issues:
        return ValidationResult(
            check_name="numeric_range",
            status=ValidationStatus.WARNING,
            message=f"Values outside expected range in column '{column}'",
            details={"column": column, "issues": issues}
        )
    return ValidationResult(
        check_name="numeric_range",
        status=ValidationStatus.PASS,
        message=f"All values in '{column}' within expected range",
        details={"column": column, "min_expected": min_value, "max_expected": max_value}
    )


def validate_categorical_values(
    df: pd.DataFrame,
    column: str,
    allowed_values: list
) -> ValidationResult:
    """Validate that categorical column only contains allowed values."""
    if column not in df.columns:
        return ValidationResult(
            check_name="categorical_values",
            status=ValidationStatus.FAIL,
            message=f"Column '{column}' not found",
            details={"column": column}
        )

    series = df[column].dropna()
    if series.empty:
        return ValidationResult(
            check_name="categorical_values",
            status=ValidationStatus.WARNING,
            message=f"Column '{column}' has no non-null values",
            details={"column": column}
        )

    invalid = series[~series.isin(allowed_values)]
    if len(invalid) > 0:
        return ValidationResult(
            check_name="categorical_values",
            status=ValidationStatus.WARNING,
            message=f"Column '{column}' contains {len(invalid)} invalid value(s)",
            details={
                "column": column,
                "invalid_values": invalid.unique().tolist(),
                "allowed_values": allowed_values
            }
        )
    return ValidationResult(
        check_name="categorical_values",
        status=ValidationStatus.PASS,
        message=f"All values in '{column}' are valid",
        details={"column": column, "allowed_values": allowed_values}
    )


def validate_row_count(
    df: pd.DataFrame,
    min_rows: Optional[int] = None,
    max_rows: Optional[int] = None
) -> ValidationResult:
    """Validate DataFrame row count is within expected bounds."""
    row_count = len(df)
    issues = []

    if min_rows is not None and row_count < min_rows:
        issues.append(f"Row count {row_count} below minimum {min_rows}")
    if max_rows is not None and row_count > max_rows:
        issues.append(f"Row count {row_count} exceeds maximum {max_rows}")

    if issues:
        return ValidationResult(
            check_name="row_count",
            status=ValidationStatus.FAIL,
            message="; ".join(issues),
            details={"row_count": row_count, "min_rows": min_rows, "max_rows": max_rows}
        )
    return ValidationResult(
        check_name="row_count",
        status=ValidationStatus.PASS,
        message=f"Row count {row_count} within expected bounds",
        details={"row_count": row_count, "min_rows": min_rows, "max_rows": max_rows}
    )


def validate_no_unexpected_changes(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    allow_row_removal: bool = True,
    allow_column_changes: bool = True
) -> ValidationResult:
    """Validate that no unexpected destructive transformations occurred."""
    issues = []

    # Check row count
    if not allow_row_removal and len(processed_df) < len(original_df):
        issues.append(f"Rows removed unexpectedly: {len(original_df)} -> {len(processed_df)}")

    # Check column count
    if not allow_column_changes and len(processed_df.columns) != len(original_df.columns):
        issues.append(f"Column count changed: {len(original_df.columns)} -> {len(processed_df.columns)}")

    # Check for columns removed
    removed_cols = set(original_df.columns) - set(processed_df.columns)
    if removed_cols and not allow_column_changes:
        issues.append(f"Columns removed: {removed_cols}")

    # Check for new columns
    new_cols = set(processed_df.columns) - set(original_df.columns)
    if new_cols and not allow_column_changes:
        issues.append(f"Unexpected new columns: {new_cols}")

    if issues:
        return ValidationResult(
            check_name="no_unexpected_changes",
            status=ValidationStatus.WARNING,
            message="Unexpected changes detected: " + "; ".join(issues),
            details={"issues": issues}
        )
    return ValidationResult(
        check_name="no_unexpected_changes",
        status=ValidationStatus.PASS,
        message="No unexpected destructive transformations",
        details={}
    )


def run_post_preprocessing_validation(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    expected_columns: Optional[list[str]] = None,
    expected_types: Optional[dict[str, str]] = None,
    numeric_ranges: Optional[dict[str, dict]] = None,
    categorical_values: Optional[dict[str, list]] = None,
    check_missing: bool = True,
    check_duplicates: bool = True,
    allow_row_removal: bool = True,
    allow_column_changes: bool = True
) -> ValidationReport:
    """
    Run comprehensive post-preprocessing validation.

    Args:
        original_df: Original DataFrame before preprocessing.
        processed_df: DataFrame after preprocessing.
        expected_columns: List of columns that must exist.
        expected_types: Dict mapping column names to expected types.
        numeric_ranges: Dict mapping column names to {"min": x, "max": y}.
        categorical_values: Dict mapping column names to allowed values.
        check_missing: Whether to check for missing values.
        check_duplicates: Whether to check for duplicates.
        allow_row_removal: Whether row removal is expected.
        allow_column_changes: Whether column changes are expected.

    Returns:
        ValidationReport with overall status and individual results.
    """
    results = []

    # Check expected columns
    if expected_columns:
        results.append(validate_columns_exist(processed_df, expected_columns))

    # Check missing values
    if check_missing:
        results.append(validate_no_missing_values(processed_df))

    # Check duplicates
    if check_duplicates:
        results.append(validate_no_duplicates(processed_df))

    # Check data types
    if expected_types:
        results.append(validate_data_types(processed_df, expected_types))

    # Check numeric ranges
    if numeric_ranges:
        for column, range_config in numeric_ranges.items():
            results.append(validate_numeric_range(
                processed_df,
                column,
                min_value=range_config.get("min"),
                max_value=range_config.get("max")
            ))

    # Check categorical values
    if categorical_values:
        for column, allowed in categorical_values.items():
            results.append(validate_categorical_values(processed_df, column, allowed))

    # Check row count
    results.append(validate_row_count(
        processed_df,
        min_rows=1 if len(original_df) > 0 else None
    ))

    # Check for unexpected changes
    results.append(validate_no_unexpected_changes(
        original_df,
        processed_df,
        allow_row_removal=allow_row_removal,
        allow_column_changes=allow_column_changes
    ))

    return ValidationReport(results=results)