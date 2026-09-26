from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class QualityIssueSchema(BaseModel):
    """
    Schema describing a single detected data-quality issue.
    """

    issue_type: str

    column: Optional[str] = None

    count: int = Field(ge=0)

    percentage: float = Field(
        ge=0,
        le=100
    )

    severity: str

    evidence: Dict[str, Any] = Field(
        default_factory=dict
    )


class QualityReportSchema(BaseModel):
    """
    Schema containing the complete quality report
    for a dataset.
    """

    total_issues: int = Field(ge=0)

    issues: List[QualityIssueSchema]