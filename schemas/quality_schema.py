from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from schemas.dataset_schema import ColumnSchema


class QualityIssueSchema(BaseModel):
    issue_type: str
    column: Optional[str] = None
    count: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)
    severity: str
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