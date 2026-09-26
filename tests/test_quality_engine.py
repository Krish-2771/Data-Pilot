import json

import pandas as pd

from quality_checks.quality_engine import (
    run_quality_checks,
    build_quality_report,
    build_dataset_quality_report,
)
from schemas.quality_schema import (
    QualityReportSchema,
    DatasetQualityReportSchema,
)


def create_sample_dataframe():
    return pd.DataFrame({
        "Name": [
            "Rahul",
            "Amit",
            "Priya",
            "Neha",
            "Rahul",
        ],
        "Age": [
            21,
            22,
            None,
            25,
            150,
        ],
        "Salary": [
            35000,
            42000,
            48000,
            55000,
            55000,
        ],
        "Department": [
            "IT",
            "it",
            "HR",
            "HR",
            "Finance",
        ],
    })


def test_run_quality_checks():
    df = create_sample_dataframe()

    issues = run_quality_checks(df)

    assert isinstance(issues, list)
    assert len(issues) > 0

    for issue in issues:
        assert "issue_type" in issue
        assert "count" in issue
        assert "percentage" in issue
        assert "severity" in issue
        assert "evidence" in issue


def test_quality_issue_structure():
    df = create_sample_dataframe()

    issues = run_quality_checks(df)

    for issue in issues:
        assert isinstance(issue["issue_type"], str)
        assert isinstance(issue["count"], int)
        assert isinstance(issue["percentage"], float)
        assert isinstance(issue["severity"], str)
        assert isinstance(issue["evidence"], dict)


def test_build_quality_report():
    df = create_sample_dataframe()

    report = build_quality_report(df)

    assert isinstance(report, QualityReportSchema)
    assert report.total_issues == len(report.issues)
    assert report.total_issues > 0


def test_build_dataset_quality_report():
    df = create_sample_dataframe()

    report = build_dataset_quality_report(
        df,
        file_name="sample_dataset.csv",
    )

    assert isinstance(report, DatasetQualityReportSchema)

    assert report.file_name == "sample_dataset.csv"
    assert report.rows == 5
    assert report.columns == 4
    assert report.memory_usage_bytes > 0

    assert len(report.column_info) == 4
    assert report.total_issues == len(report.issues)
    assert report.total_issues > 0


def test_dataset_quality_report_column_information():
    df = create_sample_dataframe()

    report = build_dataset_quality_report(df)

    column_names = [
        column.name
        for column in report.column_info
    ]

    assert column_names == [
        "Name",
        "Age",
        "Salary",
        "Department",
    ]


def test_dataset_quality_report_json():
    df = create_sample_dataframe()

    report = build_dataset_quality_report(
        df,
        file_name="sample_dataset.csv",
    )

    report_json = report.model_dump_json()

    parsed = json.loads(report_json)

    assert parsed["file_name"] == "sample_dataset.csv"
    assert parsed["rows"] == 5
    assert parsed["columns"] == 4
    assert "column_info" in parsed
    assert "issues" in parsed
    assert parsed["total_issues"] == len(parsed["issues"])


def test_quality_report_does_not_contain_raw_dataframe():
    df = create_sample_dataframe()

    report = build_dataset_quality_report(df)

    report_json = report.model_dump_json()

    assert "Rahul" not in report_json
    assert "Priya" not in report_json
    assert "Ahmedabad" not in report_json