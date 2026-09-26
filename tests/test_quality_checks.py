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


def create_quality_test_dataframe():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 4],
        "Age": [21, 22, None, 25, 150],
        "Salary": [35000, 42000, 48000, 55000, 55000],
        "Department": ["IT", "it", "HR", "HR", "Finance"],
    })


def test_missing_values():
    df = create_quality_test_dataframe()

    issues = check_missing_values(df)

    assert len(issues) == 1
    assert issues[0]["issue_type"] == "missing_values"
    assert issues[0]["column"] == "Age"
    assert issues[0]["count"] == 1


def test_duplicates():
    df = create_quality_test_dataframe()

    issues = check_duplicates(df)

    assert len(issues) == 0


def test_data_types():
    df = pd.DataFrame({
        "Age": ["21", "22", "23", "24"],
        "Name": ["A", "B", "C", "D"],
    })

    issues = check_data_types(df)

    assert isinstance(issues, list)


def test_outliers():
    df = create_quality_test_dataframe()

    issues = check_outliers(df)

    age_issues = [
        issue
        for issue in issues
        if issue["column"] == "Age"
    ]

    assert len(age_issues) == 1
    assert age_issues[0]["count"] == 1


def test_categorical_consistency():
    df = create_quality_test_dataframe()

    issues = check_categorical_consistency(df)

    department_issues = [
        issue
        for issue in issues
        if issue["column"] == "Department"
    ]

    assert len(department_issues) == 1
    assert department_issues[0]["issue_type"] == (
        "categorical_inconsistency"
    )


def test_invalid_values():
    df = create_quality_test_dataframe()

    issues = check_invalid_values(df)

    age_issues = [
        issue
        for issue in issues
        if issue["column"] == "Age"
    ]

    assert len(age_issues) >= 1


def test_cardinality():
    df = create_quality_test_dataframe()

    issues = check_cardinality(df)

    assert isinstance(issues, list)


def test_constant_columns():
    df = pd.DataFrame({
        "constant": ["A", "A", "A", "A"],
        "value": [1, 2, 3, 4],
    })

    issues = check_constant_columns(df)

    assert len(issues) >= 1

    constant_issues = [
        issue
        for issue in issues
        if issue["column"] == "constant"
    ]

    assert len(constant_issues) == 1


def test_correlations():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10],
    })

    issues = check_correlations(df)

    assert isinstance(issues, list)


def test_id_detection():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "value": [10, 20, 30, 40, 50],
    })

    issues = check_id_detection(df)

    assert isinstance(issues, list)


def test_leakage():
    df = pd.DataFrame({
        "age": [20, 21, 22, 23],
        "target": [1, 0, 1, 0],
    })

    issues = check_leakage(df)

    assert len(issues) >= 1

    leakage_issues = [
        issue
        for issue in issues
        if issue["issue_type"] == "possible_data_leakage"
    ]

    assert len(leakage_issues) == 1