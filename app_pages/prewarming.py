"""
Pre-warming module for Data-Pilot Streamlit application.

This module provides cached resources and data loading with background refresh
to ensure fast initial load and responsive UI.
"""

import os
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ============================================================
# Cached Resources (API clients, models, heavy objects)
# ============================================================

@st.cache_resource(ttl="1h", show_spinner="Initializing NVIDIA NIM client...")
def get_nim_client():
    """Get cached NVIDIA NIM client with pre-warmed connection."""
    from ai.client import NIMClient
    from ai.exceptions import MissingAPIKeyError

    try:
        client = NIMClient.from_env()
        # Pre-warm: test connection in background
        client.test_connection()
        return client
    except MissingAPIKeyError:
        return None
    except Exception as e:
        st.warning(f"NIM client initialization warning: {e}")
        return None


@st.cache_resource(ttl="1h", show_spinner="Loading AI agent...")
def get_data_pilot_agent():
    """Get cached DataPilotAgent instance."""
    from ai.agent import DataPilotAgent

    client = get_nim_client()
    if client is None:
        return None
    return DataPilotAgent(client=client)


# ============================================================
# Cached Data Loading (with background refresh)
# ============================================================

@st.cache_data(ttl="10m", refresh_mode="background", show_spinner="Loading sample dataset...")
def load_sample_dataset() -> pd.DataFrame:
    """Load and cache the sample dataset with background refresh."""
    sample_path = Path(__file__).parent.parent / "data" / "sample" / "sample_dataset.csv"
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        return df
    return pd.DataFrame()


@st.cache_data(ttl="5m", refresh_mode="background", show_spinner="Loading uploaded dataset...")
def load_uploaded_dataset(
    file_bytes: bytes, file_name: str, sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """Load and cache an uploaded dataset."""
    import io

    extension = Path(file_name).suffix.lower()
    if extension == ".csv":
        return _make_column_names_unique(pd.read_csv(io.BytesIO(file_bytes)))
    elif extension in (".xlsx", ".xls"):
        return _make_column_names_unique(
            pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
        )
    else:
        raise ValueError(f"Unsupported file format: {file_name}")


def _make_column_names_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Disambiguate duplicate headers while retaining every source column."""
    original_names = list(df.columns)
    reserved = {str(name) for name in original_names}
    used: set[str] = set()
    counts: dict[str, int] = {}
    unique_names: list[str] = []

    for original_name in original_names:
        name = str(original_name)
        counts[name] = counts.get(name, 0) + 1
        candidate = name
        suffix = counts[name]
        while candidate in used:
            candidate = f"{name}__{suffix}"
            suffix += 1
        while candidate in reserved and candidate not in used and candidate != name:
            candidate = f"{name}__{suffix}"
            suffix += 1
        unique_names.append(candidate)
        used.add(candidate)

    if unique_names != original_names:
        df = df.copy()
        df.columns = unique_names
    return df


@st.cache_data(ttl="5m", show_spinner=False)
def get_excel_sheet_names(file_bytes: bytes) -> list[str]:
    """Return workbook sheet names so users can choose the data sheet."""
    import io

    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


@st.cache_data(ttl="5m", refresh_mode="background", show_spinner="Profiling dataset...")
def profile_dataset_cached(df: pd.DataFrame) -> dict:
    """Cache dataset profiling results."""
    from profiler.dataset_profile import profile_dataset_as_schema

    profile = profile_dataset_as_schema(df)
    return profile.model_dump(exclude_none=True)


@st.cache_data(ttl="5m", refresh_mode="background", show_spinner="Running quality checks...")
def run_quality_checks_cached(
    df: pd.DataFrame,
    outlier_method: str = "iqr",
    outlier_multiplier: float = 1.5,
    outlier_threshold: float = 3.0,
) -> dict:
    """Cache quality check results."""
    from quality_checks.quality_engine import build_dataset_quality_report

    report = build_dataset_quality_report(
        df,
        outlier_method=outlier_method,
        outlier_multiplier=outlier_multiplier,
        outlier_threshold=outlier_threshold,
    )
    return report.model_dump(exclude_none=True)


@st.cache_data(ttl="5m", refresh_mode="background", show_spinner="Generating AI recommendations...")
def generate_ai_recommendations_cached(
    quality_report_dict: dict, use_compact_prompt: bool = True
) -> dict:
    """Cache AI recommendations."""
    from ai.recommender import generate_recommendations
    from schemas.quality_schema import DatasetQualityReportSchema
    from ai.client import NIMClient
    from ai.exceptions import MissingAPIKeyError, APIError

    # Reconstruct quality report schema
    quality_report = DatasetQualityReportSchema(**quality_report_dict)

    # Get client
    try:
        client = NIMClient.from_env()
    except MissingAPIKeyError:
        return {"error": "NVIDIA_API_KEY not configured"}
    except Exception as e:
        return {"error": f"Failed to initialize NIM client: {e}"}

    try:
        recommendations = generate_recommendations(
            quality_report,
            client=client,
            use_compact_prompt=use_compact_prompt,
        )
        return recommendations.model_dump(exclude_none=True)
    except APIError as e:
        return {"error": f"API error: {e}"}
    except Exception as e:
        return {"error": f"Failed to generate recommendations: {e}"}


def run_preprocessing_pipeline_cached(df: pd.DataFrame, actions: list) -> dict:
    """Run preprocessing without caching its DataFrame and action-log result."""
    from preprocessing.pipeline import run_preprocessing_pipeline

    processed_df, summary = run_preprocessing_pipeline(df, actions)
    return {
        "processed_df": processed_df,
        "summary": summary,
    }


@st.cache_data(ttl="5m", refresh_mode="background", show_spinner="Running validation...")
def run_validation_cached(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    check_missing: bool = True,
    check_duplicates: bool = True,
    allow_row_removal: bool = True,
    allow_column_changes: bool = True,
) -> dict:
    """Cache validation results."""
    from preprocessing.validation import run_post_preprocessing_validation

    validation_report = run_post_preprocessing_validation(
        original_df=original_df,
        processed_df=processed_df,
        check_missing=check_missing,
        check_duplicates=check_duplicates,
        allow_row_removal=allow_row_removal,
        allow_column_changes=allow_column_changes,
    )
    return validation_report.to_dict()


# ============================================================
# Pre-warming Functions
# ============================================================

def prewarm_resources():
    """Pre-warm all cached resources in the background."""
    # This function can be called to trigger background pre-warming
    # The actual caching happens when the cached functions are first called
    pass


def get_cached_sample_df() -> pd.DataFrame:
    """Get the cached sample dataframe."""
    return load_sample_dataset()


def is_prewarm_ready() -> bool:
    """Check if pre-warming is complete."""
    return st.session_state.get("prewarm_ready", False)


def mark_prewarm_ready():
    """Mark pre-warming as complete."""
    st.session_state.prewarm_ready = True


# ============================================================
# Utility Functions
# ============================================================

def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def get_severity_color(severity: str) -> str:
    """Get color for severity level."""
    colors = {
        "critical": "red",
        "high": "orange",
        "medium": "yellow",
        "low": "blue",
        "info": "green",
    }
    return colors.get(severity.lower(), "gray")


def display_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Display a styled metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


# ============================================================
# Session State Management
# ============================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "uploaded_df": None,
        "uploaded_file_name": None,
        "uploaded_file_fingerprint": None,
        "uploaded_sheet_name": None,
        "quality_report": None,
        "dataset_profile": None,
        "ai_recommendations": None,
        "ai_rec_failed": False,
        "approved_actions": [],
        "processed_df": None,
        "preprocessing_summary": None,
        "validation_report": None,
        "validation_config": None,
        "show_validation_download": False,
        "show_full_report_download": False,
        "last_full_report": None,
        "last_export_name": None,
        "current_step": 0,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_session_state():
    """Reset session state for new analysis."""
    keys_to_reset = [
        "uploaded_df",
        "uploaded_file_name",
        "uploaded_file_fingerprint",
        "uploaded_sheet_name",
        "quality_report",
        "dataset_profile",
        "ai_recommendations",
        "ai_rec_failed",
        "approved_actions",
        "processed_df",
        "preprocessing_summary",
        "validation_report",
        "validation_config",
        "show_validation_download",
        "show_full_report_download",
        "last_full_report",
        "last_export_name",
        "current_step",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()


def get_current_dataset() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Get the current dataset (uploaded or sample)."""
    if st.session_state.get("uploaded_df") is not None:
        return st.session_state.uploaded_df, st.session_state.uploaded_file_name
    return load_sample_dataset(), "sample_dataset.csv"
