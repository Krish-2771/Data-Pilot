import pandas as pd
from typing import Optional

from profiler.dataset_profile import profile_dataset, profile_dataset_as_schema

from quality_checks.missing_values import check_missing_values
from quality_checks.duplicates import check_duplicates
from quality_checks.data_types import check_data_types
from quality_checks.outliers import check_outliers
from quality_checks.categorical import check_categorical_consistency
from quality_checks.invalid_values import check_invalid_values
from quality_checks.cardinality import check_cardinality
from quality_checks.constants import check_constant_columns
from quality_checks.correlations import check_correlations
from quality_checks.id_detection import check_id_detection
from quality_checks.leakage import check_leakage
from quality_checks.string_checks import check_empty_strings, check_whitespace_only_strings
from quality_checks.datetime_checks import check_invalid_datetimes

from schemas.quality_schema import (
    QualityReportSchema,
    DatasetQualityReportSchema,
    QualityIssueSchema,
)


def run_quality_checks(
    df: pd.DataFrame,
    outlier_method: str = "iqr",
    outlier_multiplier: float = 1.5,
    outlier_threshold: float = 3.0
) -> list[dict]:
    issues = []

    checks = [
        check_missing_values,
        check_duplicates,
        check_data_types,
        lambda d: check_outliers(d, method=outlier_method, multiplier=outlier_multiplier, threshold=outlier_threshold),
        check_categorical_consistency,
        check_invalid_values,
        check_cardinality,
        check_constant_columns,
        check_correlations,
        check_id_detection,
        check_leakage,
        check_empty_strings,
        check_whitespace_only_strings,
        check_invalid_datetimes,
    ]

    for check in checks:
        results = check(df)

        if results:
            issues.extend(results)

    return issues


def _categorize_issues(issues: list[dict]) -> dict[str, list[QualityIssueSchema]]:
    """Categorize issues by type for structured access."""
    categories = {
        "missing_value_issues": [],
        "duplicate_issues": [],
        "outlier_issues": [],
        "type_issues": [],
        "categorical_consistency_issues": [],
        "invalid_value_issues": [],
        "cardinality_issues": [],
        "constant_column_issues": [],
        "correlation_issues": [],
        "id_detection_issues": [],
        "leakage_issues": [],
        "string_issues": [],
    }

    for issue in issues:
        issue_schema = QualityIssueSchema(**issue)
        issue_type = issue.get("issue_type", "")

        if issue_type == "missing_values":
            categories["missing_value_issues"].append(issue_schema)
        elif issue_type == "duplicate_rows":
            categories["duplicate_issues"].append(issue_schema)
        elif issue_type in ("outliers_iqr", "outliers_zscore", "outliers"):
            categories["outlier_issues"].append(issue_schema)
        elif issue_type in ("possible_numeric_type_mismatch", "possible_datetime_type_mismatch", "mixed_data_types"):
            categories["type_issues"].append(issue_schema)
        elif issue_type == "categorical_inconsistency":
            categories["categorical_consistency_issues"].append(issue_schema)
        elif issue_type in ("invalid_age_values", "negative_numeric_values"):
            categories["invalid_value_issues"].append(issue_schema)
        elif issue_type in ("high_cardinality", "low_cardinality"):
            categories["cardinality_issues"].append(issue_schema)
        elif issue_type in ("constant_column", "near_constant_column"):
            categories["constant_column_issues"].append(issue_schema)
        elif issue_type == "strong_correlation":
            categories["correlation_issues"].append(issue_schema)
        elif issue_type == "possible_identifier":
            categories["id_detection_issues"].append(issue_schema)
        elif issue_type in ("possible_data_leakage", "duplicate_information"):
            categories["leakage_issues"].append(issue_schema)
        elif issue_type in ("empty_strings", "whitespace_only_strings"):
            categories["string_issues"].append(issue_schema)
        elif issue_type == "invalid_datetime_values":
            categories["invalid_value_issues"].append(issue_schema)

    return categories


def _generate_overall_summary(issues: list[dict], df: pd.DataFrame) -> str:
    """Generate a human-readable overall summary of the quality report."""
    total_issues = len(issues)
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for issue in issues:
        severity = issue.get("severity", "low")
        if severity in severity_counts:
            severity_counts[severity] += 1

    parts = [f"Dataset has {total_issues} total quality issues detected."]
    if severity_counts["critical"] > 0:
        parts.append(f"{severity_counts['critical']} critical issue(s).")
    if severity_counts["high"] > 0:
        parts.append(f"{severity_counts['high']} high severity issue(s).")
    if severity_counts["medium"] > 0:
        parts.append(f"{severity_counts['medium']} medium severity issue(s).")
    if severity_counts["low"] > 0:
        parts.append(f"{severity_counts['low']} low severity issue(s).")

    return " ".join(parts)


def build_quality_report(
    df: pd.DataFrame,
    outlier_method: str = "iqr",
    outlier_multiplier: float = 1.5,
    outlier_threshold: float = 3.0
) -> QualityReportSchema:

    issues = run_quality_checks(
        df,
        outlier_method=outlier_method,
        outlier_multiplier=outlier_multiplier,
        outlier_threshold=outlier_threshold
    )

    report = QualityReportSchema(
        total_issues=len(issues),
        issues=[QualityIssueSchema(**issue) for issue in issues]
    )

    return report


def build_dataset_quality_report(
    df: pd.DataFrame,
    file_name: Optional[str] = None,
    outlier_method: str = "iqr",
    outlier_multiplier: float = 1.5,
    outlier_threshold: float = 3.0
) -> DatasetQualityReportSchema:

    profile = profile_dataset_as_schema(df)
    issues = run_quality_checks(
        df,
        outlier_method=outlier_method,
        outlier_multiplier=outlier_multiplier,
        outlier_threshold=outlier_threshold
    )

    categories = _categorize_issues(issues)
    overall_summary = _generate_overall_summary(issues, df)

    report = DatasetQualityReportSchema(
        file_name=file_name,
        rows=profile.rows,
        columns=profile.columns,
        memory_usage_bytes=profile.memory_usage_bytes,
        column_info=profile.column_info,
        total_issues=len(issues),
        issues=[QualityIssueSchema(**issue) for issue in issues],
        **categories,
        overall_summary=overall_summary
    )

    return report