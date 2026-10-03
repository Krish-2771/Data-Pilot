"""
Upload Dataset Page

Allows users to upload a CSV or Excel file, or use the built-in sample dataset.
Pre-warms the sample dataset for instant access.
"""

import streamlit as st
import pandas as pd
import hashlib
from typing import Optional

from app_pages.prewarming import (
    init_session_state,
    get_cached_sample_df,
    load_uploaded_dataset,
    get_excel_sheet_names,
    reset_session_state,
    format_bytes,
)
from app_pages.navigation_registry import get_page


def upload_page():
    """Main upload page function."""
    # Initialize session state
    init_session_state()

    # Page header
    st.title("📤 Upload Dataset")
    st.caption("Upload a CSV or Excel file, or use the built-in sample dataset to get started.")

    # Pre-warm the sample dataset in background
    sample_df = get_cached_sample_df()

    # Check if we have an uploaded file in session state
    has_uploaded_file = st.session_state.get("uploaded_df") is not None and st.session_state.get("uploaded_file_name") != "sample_dataset.csv"

    # Sidebar: Dataset selection
    with st.sidebar:
        st.header("Dataset Source")

        # Default to sample only if no uploaded file exists
        default_use_sample = not has_uploaded_file

        use_sample = st.toggle(
            "Use sample dataset",
            value=default_use_sample,
            help="Use the built-in sample dataset for quick testing",
            disabled=has_uploaded_file,  # Disable if user uploaded a file
        )

        if has_uploaded_file:
            st.success(f"📁 Using uploaded file: {st.session_state.uploaded_file_name}")
            # Force use_sample to False when file is uploaded
            use_sample = False
        elif use_sample:
            st.info("📊 Using sample dataset (10 rows, 5 columns)")

    # Handle dataset selection logic
    if use_sample and not has_uploaded_file:
        # User explicitly chose sample dataset
        st.session_state.uploaded_df = sample_df.copy()
        st.session_state.uploaded_file_name = "sample_dataset.csv"
        display_sample_dataset(sample_df)
    elif has_uploaded_file:
        # User has uploaded a file - show it
        current_df = st.session_state.uploaded_df
        current_name = st.session_state.uploaded_file_name
        display_dataset_preview(current_df, current_name)

        # Also show upload widget for replacing the file
        st.divider()
        st.subheader("🔄 Replace Dataset")
        display_upload_widget()
    else:
        # No file uploaded yet, no sample selected - show upload widget
        display_upload_widget()

    # Show proceed button if a dataset is loaded
    current_df, file_name = get_current_dataset()
    if current_df is not None:
        if not has_uploaded_file and not (use_sample and not has_uploaded_file):
            # Show preview for sample dataset if not already shown above
            display_dataset_preview(current_df, file_name)
        show_proceed_button()


def show_proceed_button():
    """Show the 'Proceed to Quality Analysis' button at the bottom."""
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("✅ Dataset loaded. Navigate to **Quality Analysis** to continue.")
    with col2:
        if st.button("Proceed to Quality Analysis →", type="primary", use_container_width=True, key="proceed_to_quality_analysis"):
            st.switch_page(get_page("Quality Analysis"))


def display_sample_dataset(df: pd.DataFrame):
    """Display sample dataset info."""
    st.subheader("Sample Dataset")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory", format_bytes(df.memory_usage(deep=True).sum()))
    with col4:
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("Missing %", f"{missing_pct:.1f}%")

    st.caption("This sample dataset contains intentional data quality issues for demonstration:")


def display_upload_widget():
    """Display file upload widget."""
    st.subheader("Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Maximum file size: 200MB. Supported formats: CSV, Excel (.xlsx, .xls)",
    )

    if uploaded_file is not None:
        with st.spinner("Loading dataset..."):
            try:
                file_bytes = uploaded_file.getvalue()
                extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
                selected_sheet = None
                if extension in {"xlsx", "xls"}:
                    sheet_names = get_excel_sheet_names(file_bytes)
                    if not sheet_names:
                        raise ValueError("The workbook contains no worksheets.")
                    file_fingerprint = hashlib.sha256(file_bytes).hexdigest()
                    selected_sheet = st.selectbox(
                        "Worksheet to analyze",
                        sheet_names,
                        key=f"excel_sheet_{file_fingerprint}",
                        help="Only the selected worksheet will be analyzed and exported.",
                    )
                    if not st.button("Load selected worksheet", key=f"load_sheet_{file_fingerprint}"):
                        return

                fingerprint = hashlib.sha256(
                    file_bytes + (selected_sheet or "").encode("utf-8")
                ).hexdigest()
                if st.session_state.get("uploaded_file_fingerprint") != fingerprint:
                    reset_session_state()
                df = load_uploaded_dataset(file_bytes, uploaded_file.name, selected_sheet)
                st.session_state.uploaded_df = df
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.uploaded_file_fingerprint = fingerprint
                st.session_state.uploaded_sheet_name = selected_sheet
                st.success(f"✅ Loaded {uploaded_file.name} ({len(df)} rows, {len(df.columns)} columns)")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to load dataset: {e}")


def display_dataset_preview(df: pd.DataFrame, file_name: str):
    """Display dataset preview with column info."""
    st.divider()
    st.subheader(f"📋 Dataset Preview: {file_name}")

    # Dataset info cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory", format_bytes(df.memory_usage(deep=True).sum()))
    with col4:
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("Missing %", f"{missing_pct:.1f}%")

    # Column info table
    st.write("**Column Information**")
    col_info = []
    for col in df.columns:
        col_data = df[col]
        col_info.append({
            "Column": col,
            "Type": str(col_data.dtype),
            "Non-Null": col_data.notna().sum(),
            "Null": col_data.isna().sum(),
            "Null %": f"{(col_data.isna().sum() / len(df) * 100):.1f}%",
            "Unique": col_data.nunique(),
        })

    col_info_df = pd.DataFrame(col_info)
    st.dataframe(col_info_df, hide_index=True, width="stretch")

    # Data preview
    st.write("**Data Preview (first 100 rows)**")
    st.dataframe(df.head(100), hide_index=True, width="stretch")


def get_current_dataset() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Get the current dataset from session state."""
    df = st.session_state.get("uploaded_df")
    file_name = st.session_state.get("uploaded_file_name")
    return df, file_name
