"""
Data-Pilot AI Agent - High-level interface for AI-powered data quality analysis.

This module provides the DataPilotAgent class that orchestrates the full
AI-powered workflow: profiling → quality checks → AI recommendations.
"""

from typing import Optional, Union
import pandas as pd

from ai.client import NIMClient
from ai.recommender import generate_recommendations
from ai.schemas import (
    AIRecommendationResponse,
    PreprocessingAction,
    RecommendationSchema,
)
from ai.exceptions import DataPilotAIError, MissingAPIKeyError
from schemas.quality_schema import DatasetQualityReportSchema
from quality_checks.quality_engine import build_dataset_quality_report


class DataPilotAgent:
    """
    High-level AI agent for Data-Pilot data quality analysis and recommendations.

    This agent provides a simple interface to:
    1. Analyze dataset quality (deterministic profiling + quality checks)
    2. Generate AI-powered preprocessing recommendations
    3. Explain specific quality issues
    """

    def __init__(
        self,
        client: Optional[NIMClient] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout: float = 60.0,
    ):
        """
        Initialize the DataPilotAgent.

        Args:
            client: Pre-configured NIMClient instance (optional).
            api_key: NVIDIA API key. If None, reads from environment.
            model: Model name. If None, uses default.
            base_url: NIM base URL. If None, uses default.
            temperature: Generation temperature.
            max_tokens: Max tokens for response.
            timeout: Request timeout in seconds.

        Raises:
            MissingAPIKeyError: If no API key is available.
        """
        if client is not None:
            self._client = client
        else:
            self._client = NIMClient.from_env()

    def analyze_quality_report(
        self,
        quality_report: DatasetQualityReportSchema,
    ) -> AIRecommendationResponse:
        """
        Analyze an existing quality report and generate AI recommendations.

        Args:
            quality_report: Structured quality report from quality engine.

        Returns:
            AIRecommendationResponse with structured recommendations.

        Raises:
            DataPilotAIError: If AI analysis fails.
        """
        return generate_recommendations(quality_report, client=self._client)

    def recommend_preprocessing(
        self,
        df: pd.DataFrame,
    ) -> AIRecommendationResponse:
        """
        Full pipeline: profile → quality checks → AI recommendations.

        This is the main entry point for end-to-end analysis.

        Args:
            df: Input DataFrame to analyze.

        Returns:
            AIRecommendationResponse with structured recommendations.

        Raises:
            DataPilotAIError: If any step fails.
        """
        # Generate quality report
        quality_report = build_dataset_quality_report(df)

        # Generate AI recommendations
        return self.analyze_quality_report(quality_report)

    def explain_issue(
        self,
        quality_report: DatasetQualityReportSchema,
        issue_type: str,
        column: Optional[str] = None,
    ) -> str:
        """
        Get AI explanation for a specific quality issue.

        Args:
            quality_report: Structured quality report.
            issue_type: Type of issue (e.g., "missing_values", "outliers_iqr").
            column: Optional specific column to explain.

        Returns:
            Human-readable explanation from AI.
        """
        # Extract relevant issue information
        issues = quality_report.issues

        if column:
            relevant_issues = [
                issue for issue in issues
                if issue.issue_type == issue_type and issue.column == column
            ]
        else:
            relevant_issues = [
                issue for issue in issues
                if issue.issue_type == issue_type
            ]

        if not relevant_issues:
            return f"No issues of type '{issue_type}' found for column '{column}'."

        # Build context for explanation
        issue_details = []
        for issue in relevant_issues:
            detail = f"- Column: {issue.column}, Severity: {issue.severity}, Count: {issue.count}"
            if issue.evidence:
                detail += f", Evidence: {issue.evidence}"
            issue_details.append(detail)

        context = f"""
Issue Type: {issue_type}
Column: {column or 'All columns'}
Found Issues:
{chr(10).join(issue_details)}

Please explain this issue in plain language and suggest what preprocessing
actions from the following list might be appropriate:
{', '.join(PreprocessingAction.all_actions())}

Keep the explanation concise and actionable.
"""

        messages = [
            {"role": "system", "content": "You are a data quality expert explaining issues to a data scientist."},
            {"role": "user", "content": context},
        ]

        try:
            response = self._client.chat_completion(messages=messages)
            return response
        except Exception as e:
            return f"Failed to get explanation: {e}"

    def get_supported_actions(self) -> list[str]:
        """Return list of all supported preprocessing actions."""
        return PreprocessingAction.all_actions()

    def test_connection(self) -> bool:
        """Test connection to NVIDIA NIM."""
        return self._client.test_connection()

    @property
    def client(self) -> NIMClient:
        """Access the underlying NIM client."""
        return self._client