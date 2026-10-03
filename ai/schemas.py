"""
Pydantic schemas for AI agent structured output.

These schemas define the structured interface between the NVIDIA NIM model
and the Data-Pilot preprocessing pipeline.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal
from enum import Enum


class PreprocessingAction(str, Enum):
    """Valid preprocessing actions supported by Data-Pilot pipeline."""

    # Missing value handling
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    FILL_UNKNOWN = "fill_unknown"
    DROP_ROWS = "drop_rows"

    # Duplicate handling
    DROP_DUPLICATES = "drop_duplicates"

    # Outlier handling
    REMOVE_OUTLIERS = "remove_outliers"
    CLIP_OUTLIERS = "clip_outliers"

    # Categorical handling
    NORMALIZE_CATEGORICAL_VALUES = "normalize_categorical_values"
    LABEL_ENCODING = "label_encoding"
    ONE_HOT_ENCODING = "one_hot_encoding"

    # Scaling
    STANDARD = "standard"
    MINMAX = "minmax"

    # String cleaning
    TRIM_WHITESPACE = "trim_whitespace"
    EMPTY_STRINGS_TO_MISSING = "empty_strings_to_missing"
    NORMALIZE_CATEGORICAL_CASE = "normalize_categorical_case"

    # Type conversion
    CONVERT_TO_NUMERIC = "convert_to_numeric"
    CONVERT_TO_DATETIME = "convert_to_datetime"
    CONVERT_TO_CATEGORICAL = "convert_to_categorical"

    @classmethod
    def all_actions(cls) -> List[str]:
        """Return list of all valid action strings."""
        return [action.value for action in cls]


class RecommendationSchema(BaseModel):
    """
    Single preprocessing recommendation from the AI agent.

    Attributes:
        action: The preprocessing action to apply (must be a valid action).
        columns: List of column names to apply the action to.
        reason: Human-readable explanation for this recommendation.
        confidence: Confidence score between 0.0 and 1.0.
        requires_approval: Whether this action requires user approval (default True).
        parameters: Optional additional parameters for the action.
    """

    action: PreprocessingAction
    columns: List[str] = Field(default_factory=list)
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_approval: bool = True
    parameters: dict = Field(default_factory=dict)

    @field_validator("columns", mode="before")
    @classmethod
    def validate_columns(cls, v):
        """Ensure columns is a list of strings."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v


class AIRecommendationResponse(BaseModel):
    """
    Complete structured response from the AI recommendation engine.

    Attributes:
        summary: High-level summary of the dataset quality and recommendations.
        recommendations: List of preprocessing recommendations.
        reasoning: Detailed reasoning for the overall recommendation strategy.
    """

    summary: str
    recommendations: List[RecommendationSchema]
    reasoning: str = ""

    @property
    def has_recommendations(self) -> bool:
        """Return True if there are any recommendations."""
        return len(self.recommendations) > 0

    @property
    def action_count(self) -> int:
        """Return total number of recommended actions."""
        return len(self.recommendations)


class AISystemPromptConfig(BaseModel):
    """Configuration for the AI system prompt."""

    model_name: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=100, le=8000)
    base_url: str = "https://integrate.api.nvidia.com/v1"