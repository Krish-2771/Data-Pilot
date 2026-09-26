from pydantic import BaseModel, Field
from typing import List, Optional


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


class DatasetSchema(BaseModel):
    """
    Schema describing the overall dataset.
    """

    file_name: Optional[str] = None

    rows: int = Field(ge=0)
    columns: int = Field(ge=0)

    memory_usage_bytes: int = Field(ge=0)

    column_info: List[ColumnSchema]