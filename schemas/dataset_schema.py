from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union


class ColumnStatisticsSchema(BaseModel):
    """Schema for column-specific statistics."""

    # Numeric statistics
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    iqr: Optional[float] = None
    mode: Optional[Union[float, str]] = None
    mode_frequency: Optional[int] = None

    # Categorical statistics
    top_values: Optional[Dict[str, int]] = None

    # Datetime statistics
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    range_days: Optional[float] = None

    # Common
    count: Optional[int] = None
    unique_count: Optional[int] = None


class ColumnSchema(BaseModel):
    """
    Schema describing a single dataset column.
    """

    name: str
    data_type: str
    column_type: str
    non_null_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    missing_percentage: float = Field(
        ge=0,
        le=100
    )
    unique_count: int = Field(ge=0)
    statistics: Optional[ColumnStatisticsSchema] = None


class DuplicateRowsSchema(BaseModel):
    """Schema for duplicate row information."""

    total_rows: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    duplicate_percentage: float = Field(ge=0, le=100)


class OverallMissingSchema(BaseModel):
    """Schema for overall missing value information."""

    total_cells: int = Field(ge=0)
    missing_cells: int = Field(ge=0)
    missing_percentage: float = Field(ge=0, le=100)


class DatasetSchema(BaseModel):
    """
    Schema describing the overall dataset.
    """

    file_name: Optional[str] = None

    rows: int = Field(ge=0)
    columns: int = Field(ge=0)

    memory_usage_bytes: int = Field(ge=0)

    duplicate_rows: Optional[DuplicateRowsSchema] = None
    overall_missing: Optional[OverallMissingSchema] = None

    column_info: List[ColumnSchema]