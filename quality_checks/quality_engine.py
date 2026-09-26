
import pandas as pd

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

from schemas.quality_schema import QualityReportSchema


def run_quality_checks(df: pd.DataFrame) -> list[dict]:
    issues = []

    checks = [
        check_missing_values,
        check_duplicates,
        check_data_types,
        check_outliers,
        check_categorical_consistency,
        check_invalid_values,
        check_cardinality,
        check_constant_columns,
        check_correlations,
        check_id_detection,
        check_leakage,
    ]

    for check in checks:
        results = check(df)

        if results:
            issues.extend(results)

    return issues


def build_quality_report(df: pd.DataFrame) -> QualityReportSchema:
    issues = run_quality_checks(df)

    report = QualityReportSchema(
        total_issues=len(issues),
        issues=issues
    )

    return report