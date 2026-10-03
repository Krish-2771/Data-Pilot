"""
Validation & Results Page

Runs comprehensive post-preprocessing validation and displays final results.
Allows downloading the processed dataset.
"""

import streamlit as st
import pandas as pd
import json
from typing import Optional
from io import BytesIO

from app_pages.prewarming import (
    init_session_state,
    get_current_dataset,
    run_validation_cached,
    format_bytes,
    get_severity_color,
)
from app_pages.navigation_registry import get_page


def validation_page():
    """Main validation page function."""
    init_session_state()

    # Page header
    st.title("✅ Validation & Results")
    st.caption("Comprehensive post-preprocessing validation and final dataset export.")

    # Get datasets
    original_df, file_name = get_current_dataset()
    processed_df = st.session_state.get("processed_df")
    summary = st.session_state.get("preprocessing_summary")

    if original_df is None:
        st.warning("⚠️ No dataset loaded. Please go to Upload Dataset first.")
        if st.button("← Go to Upload", use_container_width=True):
            st.switch_page(get_page("Upload Dataset"))
        return

    if processed_df is None:
        st.warning("⚠️ No processed dataset. Please run the Preprocessing Pipeline first.")
        if st.button("← Go to Preprocessing Pipeline", use_container_width=True):
            st.switch_page(get_page("Preprocessing Pipeline"))
        return

    # Sidebar: Validation options
    with st.sidebar:
        st.header("Validation Options")

        check_missing = st.checkbox("Check missing values", value=True)
        check_duplicates = st.checkbox("Check duplicates", value=True)
        allow_row_removal = st.checkbox("Allow row removal", value=True)
        allow_column_changes = st.checkbox("Allow column changes", value=True)

        validation_config = (
            check_missing,
            check_duplicates,
            allow_row_removal,
            allow_column_changes,
        )
        if st.session_state.get("validation_config") != validation_config:
            st.session_state.validation_config = validation_config
            st.session_state.validation_report = None

        run_validation = st.button(
            "🔍 Run Validation",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("validation_report") is not None,
        )

        if st.session_state.get("validation_report") is not None:
            if st.button("🔄 Re-run Validation", use_container_width=True):
                if "validation_report" in st.session_state:
                    del st.session_state.validation_report
                st.rerun()

    # Run validation
    if run_validation or st.session_state.get("validation_report") is None:
        run_validation_with_progress(original_df, processed_df, check_missing, check_duplicates, allow_row_removal, allow_column_changes)

    # Display validation results
    validation_report = st.session_state.get("validation_report")
    if validation_report:
        display_validation_results(validation_report)

    # Export section
    st.divider()
    display_export_section(processed_df, file_name, summary, validation_report)

    # Full pipeline summary
    with st.expander("📋 Full Pipeline Summary", expanded=False):
        display_full_pipeline_summary(original_df, processed_df, summary, validation_report)


def run_validation_with_progress(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    check_missing: bool,
    check_duplicates: bool,
    allow_row_removal: bool,
    allow_column_changes: bool,
):
    """Run validation with progress indicators."""
    with st.status("Running post-preprocessing validation...", expanded=True) as status:
        try:
            st.write("🔍 Validating processed dataset...")

            result = run_validation_cached(
                original_df,
                processed_df,
                check_missing=check_missing,
                check_duplicates=check_duplicates,
                allow_row_removal=allow_row_removal,
                allow_column_changes=allow_column_changes,
            )

            st.session_state.validation_report = result

            status.update(label="✅ Validation complete!", state="complete")

        except Exception as e:
            status.update(label=f"❌ Validation failed: {e}", state="error")
            st.error(f"Validation failed: {e}")


def display_validation_results(validation_report: dict):
    """Display validation results with visual indicators."""
    overall_status = validation_report.get("overall_status", "UNKNOWN")
    results = validation_report.get("results", [])

    # Overall status badge
    status_colors = {
        "PASS": "green",
        "WARNING": "orange",
        "FAIL": "red",
    }
    status_color = status_colors.get(overall_status, "gray")

    st.subheader("📊 Validation Summary")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.badge(f"Overall Status: {overall_status}", color=status_color)
    with col2:
        passed = sum(1 for r in results if r.get("status") == "PASS")
        st.metric("Passed", passed)
    with col3:
        failed = sum(1 for r in results if r.get("status") == "FAIL")
        warnings = sum(1 for r in results if r.get("status") == "WARNING")
        st.metric("Failed / Warnings", f"{failed} / {warnings}")

    # Detailed results
    st.subheader("📋 Validation Checks")

    for result in results:
        status = result.get("status", "UNKNOWN")
        color = status_colors.get(status, "gray")

        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{result.get('check_name', 'Unknown').replace('_', ' ').title()}**")
                st.write(result.get("message", ""))
            with col2:
                st.badge(status, color=color)

            # Details
            details = result.get("details", {})
            if details:
                with st.expander("📎 Details", expanded=False):
                    st.json(details)


def display_export_section(
    processed_df: pd.DataFrame,
    original_file_name: str,
    summary: dict,
    validation_report: dict,
):
    """Display export/download section."""
    st.subheader("💾 Export Processed Dataset")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Generate base filename
        base_name = original_file_name.replace(".csv", "").replace(".xlsx", "").replace(".xls", "")
        export_name = st.text_input(
            "Export filename",
            value=f"{base_name}_processed",
            help="Filename without extension",
        )

    with col2:
        export_format = st.selectbox(
            "Format",
            ["CSV", "Excel (.xlsx)"],
            index=0,
        )

    # Preview
    st.write("**Preview (first 10 rows):**")
    st.dataframe(processed_df.head(10), hide_index=True, width="stretch")
    st.caption(f"{len(processed_df)} rows × {len(processed_df.columns)} columns | {format_bytes(processed_df.memory_usage(deep=True).sum())}")

    # Download buttons
    col1, col2 = st.columns(2)

    with col1:
        if export_format == "CSV":
            csv_data = processed_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name=f"{export_name}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        else:
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                processed_df.to_excel(writer, index=False, sheet_name="Data")
            excel_data = excel_buffer.getvalue()
            st.download_button(
                "📥 Download Excel",
                data=excel_data,
                file_name=f"{export_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
            )

    with col2:
        # Download validation report - persist visibility in session state
        if validation_report:
            st.session_state.show_validation_download = True
        elif "show_validation_download" not in st.session_state:
            st.session_state.show_validation_download = False

        if st.session_state.show_validation_download and validation_report:
            report_json = json.dumps(validation_report, indent=2, default=str)
            st.download_button(
                "📋 Download Validation Report",
                data=report_json,
                file_name=f"{export_name}_validation_report.json",
                mime="application/json",
                use_container_width=True,
            )

    # Download full pipeline report - persist visibility in session state
    if "show_full_report_download" not in st.session_state:
        st.session_state.show_full_report_download = False

    if st.button("📄 Generate Full Pipeline Report", use_container_width=True):
        st.session_state.show_full_report_download = True
        generate_full_report(original_df, processed_df, summary, validation_report, export_name)
    elif st.session_state.show_full_report_download and "last_export_name" in st.session_state:
        # Show the download button if report was already generated
        report_json = json.dumps(st.session_state.last_full_report, indent=2, default=str)
        st.download_button(
            "📥 Download Full Report (JSON)",
            data=report_json,
            file_name=f"{st.session_state.last_export_name}_full_report.json",
            mime="application/json",
            type="primary",
            use_container_width=True,
        )


def generate_full_report(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    summary: dict,
    validation_report: dict,
    export_name: str,
):
    """Generate comprehensive pipeline report."""
    from quality_checks.quality_engine import build_dataset_quality_report
    from schemas.quality_schema import DatasetQualityReportSchema

    with st.status("Generating full pipeline report...", expanded=True) as status:
        try:
            # Get quality reports for both datasets
            original_quality = build_dataset_quality_report(original_df)
            processed_quality = build_dataset_quality_report(processed_df)

            report = {
                "pipeline_summary": {
                    "original_file": export_name,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "preprocessing_summary": summary,
                    "validation_summary": validation_report,
                },
                "original_dataset": {
                    "shape": list(original_df.shape),
                    "memory_usage_bytes": int(original_df.memory_usage(deep=True).sum()),
                    "quality_report": original_quality.model_dump(exclude_none=True),
                },
                "processed_dataset": {
                    "shape": list(processed_df.shape),
                    "memory_usage_bytes": int(processed_df.memory_usage(deep=True).sum()),
                    "quality_report": processed_quality.model_dump(exclude_none=True),
                },
            }

            report_json = json.dumps(report, indent=2, default=str)

            # Store in session state for persistent download button
            st.session_state.last_full_report = report
            st.session_state.last_export_name = export_name

            st.download_button(
                "📥 Download Full Report (JSON)",
                data=report_json,
                file_name=f"{export_name}_full_report.json",
                mime="application/json",
                type="primary",
                use_container_width=True,
            )

            status.update(label="✅ Report generated!", state="complete")

        except Exception as e:
            status.update(label=f"❌ Failed: {e}", state="error")
            st.error(f"Failed to generate report: {e}")


def display_full_pipeline_summary(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    summary: dict,
    validation_report: dict,
):
    """Display complete pipeline summary."""
    st.write("**Pipeline Overview**")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Original Dataset**")
        st.metric("Rows", f"{len(original_df):,}")
        st.metric("Columns", len(original_df.columns))
        st.metric("Memory", format_bytes(original_df.memory_usage(deep=True).sum()))
        st.metric("Missing %", f"{(original_df.isna().sum().sum() / (len(original_df) * len(original_df.columns)) * 100):.1f}%")

    with col2:
        st.write("**Processed Dataset**")
        st.metric("Rows", f"{len(processed_df):,}")
        st.metric("Columns", len(processed_df.columns))
        st.metric("Memory", format_bytes(processed_df.memory_usage(deep=True).sum()))
        st.metric("Missing %", f"{(processed_df.isna().sum().sum() / (len(processed_df) * len(processed_df.columns)) * 100):.1f}%")

    # Preprocessing actions
    if summary and summary.get("action_logs"):
        st.write("**Applied Actions**")
        for i, log in enumerate(summary["action_logs"]):
            st.write(f"{i+1}. **{log.get('action', 'Unknown')}** — {log.get('columns', [])} — {log.get('rows_affected', 0)} rows affected")

    # Validation status
    if validation_report:
        overall = validation_report.get("overall_status", "UNKNOWN")
        st.write(f"**Validation Status: {overall}**")

        for result in validation_report.get("results", []):
            status = result.get("status", "UNKNOWN")
            icon = "✅" if status == "PASS" else "⚠️" if status == "WARNING" else "❌"
            st.write(f"{icon} {result.get('check_name', 'Unknown').replace('_', ' ').title()}: {result.get('message', '')}")
