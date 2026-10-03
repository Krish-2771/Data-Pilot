from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal

from schemas.dataset_schema import ColumnSchema


class SeverityLevel(str):
    """Severity levels for quality issues."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueType(str):
    """Types of quality issues."""
    MISSING_VALUES = "missing_values"
    DUPLICATE_ROWS = "duplicate_rows"
    OUTLIERS = "outliers"
    POSSIBLE_NUMERIC_TYPE_MISMATCH = "possible_numeric_type_mismatch"
    POSSIBLE_DATETIME_TYPE_MISMATCH = "possible_datetime_type_mismatch"
    MIXED_DATA_TYPES = "mixed_data_types"
    CATEGORICAL_INCONSISTENCY = "categorical_inconsistency"
    INVALID_AGE_VALUES = "invalid_age_values"
    NEGATIVE_NUMERIC_VALUES = "negative_numeric_values"
    HIGH_CARDINALITY = "high_cardinality"
    LOW_CARDINALITY = "low_cardinality"
    CONSTANT_COLUMN = "constant_column"
    NEAR_CONSTANT_COLUMN = "near_constant_column"
    STRONG_CORRELATION = "strong_correlation"
    POSSIBLE_IDENTIFIER = "possible_identifier"
    POSSIBLE_DATA_LEAKAGE = "possible_data_leakage"
    DUPLICATE_INFORMATION = "duplicate_information"
    EMPTY_STRINGS = "empty_strings"
    WHITESPACE_ONLY_STRINGS = "whitespace_only_strings"
    INVALID_DATETIME_VALUES = "invalid_datetime_values"


class QualityIssueSchema(BaseModel):
    issue_type: str
    column: Optional[str] = None
    count: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)
    severity: str
    description: Optional[str] = None
    recommended_action: Optional[str] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)


class QualityReportSchema(BaseModel):
    total_issues: int = Field(ge=0)
    issues: List[QualityIssueSchema]


class DatasetQualityReportSchema(BaseModel):
    file_name: Optional[str] = None
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    memory_usage_bytes: int = Field(ge=0)
    column_info: List[ColumnSchema]
    total_issues: int = Field(ge=0)
    issues: List[QualityIssueSchema]

    # Structured issue categories for easier access
    missing_value_issues: List[QualityIssueSchema] = Field(default_factory=list)
    duplicate_issues: List[QualityIssueSchema] = Field(default_factory=list)
    outlier_issues: List[QualityIssueSchema] = Field(default_factory=list)
    type_issues: List[QualityIssueSchema] = Field(default_factory=list)
    categorical_consistency_issues: List[QualityIssueSchema] = Field(default_factory=list)
    invalid_value_issues: List[QualityIssueSchema] = Field(default_factory=list)
    cardinality_issues: List[QualityIssueSchema] = Field(default_factory=list)
    constant_column_issues: List[QualityIssueSchema] = Field(default_factory=list)
    correlation_issues: List[QualityIssueSchema] = Field(default_factory=list)
    id_detection_issues: List[QualityIssueSchema] = Field(default_factory=list)
    leakage_issues: List[QualityIssueSchema] = Field(default_factory=list)
    string_issues: List[QualityIssueSchema] = Field(default_factory=list)

    overall_summary: Optional[str] = None