"""
Data-Pilot AI Module

NVIDIA NIM-powered AI recommendation layer for data quality analysis.
"""

from ai.agent import DataPilotAgent
from ai.client import NIMClient
from ai.recommender import generate_recommendations, generate_recommendations_sync
from ai.schemas import (
    AIRecommendationResponse,
    AISystemPromptConfig,
    PreprocessingAction,
    RecommendationSchema,
)
from ai.exceptions import (
    DataPilotAIError,
    ConfigurationError,
    MissingAPIKeyError,
    InvalidModelError,
    APIError,
    APITimeoutError,
    MalformedResponseError,
    ValidationError,
    UnsupportedActionError,
    InvalidColumnError,
    InvalidConfidenceError,
)

__all__ = [
    # Agent
    "DataPilotAgent",
    # Client
    "NIMClient",
    # Recommender
    "generate_recommendations",
    "generate_recommendations_sync",
    # Schemas
    "AIRecommendationResponse",
    "AISystemPromptConfig",
    "PreprocessingAction",
    "RecommendationSchema",
    # Exceptions
    "DataPilotAIError",
    "ConfigurationError",
    "MissingAPIKeyError",
    "InvalidModelError",
    "APIError",
    "APITimeoutError",
    "MalformedResponseError",
    "ValidationError",
    "UnsupportedActionError",
    "InvalidColumnError",
    "InvalidConfidenceError",
]

__version__ = "1.0.0"