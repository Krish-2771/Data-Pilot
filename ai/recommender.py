"""
AI recommendation engine for Data-Pilot.

This module provides the core function to generate structured preprocessing
recommendations from a quality report using NVIDIA NIM.
"""

import json
import re
from typing import Optional

from pydantic import ValidationError as PydanticValidationError

from ai.client import NIMClient
from ai.schemas import (
    AIRecommendationResponse,
    PreprocessingAction,
    RecommendationSchema,
)
from ai.exceptions import (
    MalformedResponseError,
    ValidationError,
    UnsupportedActionError,
    InvalidColumnError,
    InvalidConfidenceError,
)
from ai.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from schemas.quality_schema import DatasetQualityReportSchema


import logging

def _extract_json_from_response(response_text: str) -> dict:
    """
    Extract JSON from model response that may contain reasoning text or markdown formatting.

    The model may output chain-of-thought before the JSON or wrap JSON in code fences.
    We find valid JSON object in the response.
    """
    if not response_text or not response_text.strip():
        raise MalformedResponseError(
            "Could not extract valid JSON from model response",
            raw_response="",
        )

    text = response_text.strip()

    # 1. Direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown code fences: ```json ... ``` or ``` ... ```
    code_fence_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    code_fence_matches = list(re.finditer(code_fence_pattern, text, re.DOTALL | re.IGNORECASE))
    for match in reversed(code_fence_matches):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    # 3. Use balanced brace parsing to find all valid JSON object candidates
    candidates = []
    starts = [m.start() for m in re.finditer(r'\{', text)]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            char = text[i]
            if escape:
                escape = False
                continue
            if char == '\\' and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        candidate_str = text[start : i + 1]
                        try:
                            parsed = json.loads(candidate_str)
                            if isinstance(parsed, dict):
                                candidates.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        break

    # Prioritize candidates that have 'summary' or 'recommendations' keys
    for cand in reversed(candidates):
        if "summary" in cand or "recommendations" in cand:
            return cand

    # If no prioritized candidate, return the last candidate dictionary found
    if candidates:
        return candidates[-1]

    raise MalformedResponseError(
        "Could not extract valid JSON from model response",
        raw_response=response_text[:500],
    )



def _serialize_quality_report(report: DatasetQualityReportSchema) -> str:
    """
    Serialize quality report to JSON for the AI prompt.

    Excludes raw data and focuses on structured quality information.
    """
    # Use model_dump with exclude_none to keep JSON clean
    report_dict = report.model_dump(exclude_none=True)
    # Compact JSON without indentation for token efficiency
    return json.dumps(report_dict, separators=(',', ':'), default=str)


def _validate_recommendation(
    recommendation: dict,
    column_names: list[str],
) -> RecommendationSchema:
    """
    Validate a single recommendation against supported actions and columns.

    Args:
        recommendation: Raw recommendation dict from model response.
        column_names: List of valid column names from the quality report.

    Returns:
        Validated RecommendationSchema.

    Raises:
        ValidationError: If recommendation is invalid.
    """
    # Validate action
    action_str = recommendation.get("action")
    if not action_str:
        raise ValidationError("Recommendation missing required 'action' field")

    try:
        action = PreprocessingAction(action_str)
    except ValueError:
        valid_actions = PreprocessingAction.all_actions()
        raise UnsupportedActionError(
            f"Unsupported action '{action_str}'. Valid actions: {valid_actions}"
        )

    # Validate columns
    columns = recommendation.get("columns", [])
    if not isinstance(columns, list):
        raise ValidationError("'columns' must be a list")

    invalid_columns = [col for col in columns if col not in column_names]
    if invalid_columns:
        raise InvalidColumnError(
            f"Recommended columns not found in dataset: {invalid_columns}. "
            f"Valid columns: {column_names}"
        )

    # Validate reason
    reason = recommendation.get("reason", "")
    if not reason or not isinstance(reason, str):
        raise ValidationError("Recommendation missing or invalid 'reason'")

    # Validate confidence
    confidence = recommendation.get("confidence")
    if confidence is None:
        raise InvalidConfidenceError("Missing 'confidence' field")
    try:
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise InvalidConfidenceError(f"Confidence must be 0.0-1.0, got {confidence}")
    except (ValueError, TypeError):
        raise InvalidConfidenceError(f"Invalid confidence value: {confidence}")

    # Validate requires_approval
    requires_approval = recommendation.get("requires_approval", True)
    if not isinstance(requires_approval, bool):
        raise ValidationError("'requires_approval' must be boolean")

    # Validate parameters
    parameters = recommendation.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValidationError("'parameters' must be a dictionary")

    return RecommendationSchema(
        action=action,
        columns=columns,
        reason=reason,
        confidence=confidence,
        requires_approval=requires_approval,
        parameters=parameters,
    )


# Compact system prompt for faster inference
COMPACT_SYSTEM_PROMPT = """You are a data quality expert. Output ONLY valid JSON with keys: summary, recommendations, reasoning.
No explanations. No markdown. No text before/after JSON.

Valid actions: fill_mean, fill_median, fill_mode, fill_unknown, drop_rows, drop_duplicates, remove_outliers, clip_outliers, normalize_categorical_values, label_encoding, one_hot_encoding, standard, minmax, trim_whitespace, empty_strings_to_missing, normalize_categorical_case, convert_to_numeric, convert_to_datetime, convert_to_categorical.

Format: {"summary": "str", "recommendations": [{"action": "str", "columns": ["str"], "reason": "str", "confidence": 0.0-1.0, "requires_approval": true, "parameters": {}}], "reasoning": "str"}"""

def generate_recommendations(
    quality_report: DatasetQualityReportSchema,
    client: Optional[NIMClient] = None,
    use_compact_prompt: bool = True,  # Default to compact for speed
) -> AIRecommendationResponse:
    """
    Generate structured preprocessing recommendations from a quality report.

    This is the main entry point for the AI recommendation layer.

    Args:
        quality_report: Structured quality report from the quality engine.
        client: Optional NIMClient instance. If None, creates from environment.
        use_compact_prompt: Use shorter prompt for token efficiency (default True).

    Returns:
        Validated AIRecommendationResponse with structured recommendations.

    Raises:
        MissingAPIKeyError: If NVIDIA API key not configured.
        APIError: If NIM API call fails.
        MalformedResponseError: If model response cannot be parsed.
        ValidationError: If recommendations fail validation.
    """
    # Create client if not provided
    if client is None:
        client = NIMClient.from_env()

    # Extract column names for validation
    column_names = [col.name for col in quality_report.column_info]

    # Serialize report to JSON (compact)
    report_json = _serialize_quality_report(quality_report)

    if use_compact_prompt:
        system_prompt = COMPACT_SYSTEM_PROMPT
        user_prompt = f"Report: {report_json}\nReturn JSON recommendations."
    else:
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT_TEMPLATE.format(quality_report_json=report_json)

    # Call NIM with thinking disabled for JSON output
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response_text = client.chat_completion(
            messages=messages,
            max_tokens=2000,
            enable_thinking=False,  # Critical for nemotron JSON output
            response_format={"type": "json_object"},  # Force JSON output
        )
    except Exception as e:
        raise e  # Re-raise API errors

    # Extract JSON from response
    try:
        response_data = _extract_json_from_response(response_text)
    except MalformedResponseError:
        raise
    except Exception as e:
        raise MalformedResponseError(
            f"Failed to extract JSON from model response: {e}",
            raw_response=response_text[:500],
        )

    # Validate response structure with Pydantic
    try:
        response = AIRecommendationResponse(**response_data)
    except PydanticValidationError as e:
        raise MalformedResponseError(
            f"Model response failed schema validation: {e}",
            raw_response=response_text[:500],
        )

    # Validate each recommendation
    validated_recommendations = []
    last_validation_error = None
    for rec in response.recommendations:
        try:
            validated = _validate_recommendation(rec.model_dump(), column_names)
            validated_recommendations.append(validated)
        except ValidationError as e:
            logging.warning(f"Skipping invalid recommendation: {e}")
            last_validation_error = e

    if not validated_recommendations and response.recommendations and last_validation_error:
        raise last_validation_error

    # Return validated response
    return AIRecommendationResponse(
        summary=response.summary,
        recommendations=validated_recommendations,
        reasoning=response.reasoning,
    )


def generate_recommendations_sync(
    df,
    client: Optional[NIMClient] = None,
) -> AIRecommendationResponse:
    """
    Convenience function: profile → quality check → recommendations.

    Args:
        df: Input DataFrame.
        client: Optional NIMClient instance.

    Returns:
        AIRecommendationResponse.
    """
    from quality_checks.quality_engine import build_dataset_quality_report

    report = build_dataset_quality_report(df)
    return generate_recommendations(report, client)
