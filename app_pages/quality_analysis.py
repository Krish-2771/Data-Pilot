"""
Quality Analysis Page

Runs comprehensive quality checks on the dataset and displays results.
Uses cached profiling and quality check functions with background refresh.
"""

import streamlit as st
import pandas as pd
from typing import Optional

from app_pages.prewarming import (
    init_session_state,
    get_current_dataset,
    profile_dataset_cached,
    run_quality_checks_cached,
    format_bytes,
    get_severity_color,
)
from app_pages.navigation_registry import get_page


def quality_analysis_page():
    """Main quality analysis page function."""
    init_session_state()

    # Page header
    st.title("🔍 Quality Analysis")
    st.caption("Comprehensive dataset profiling and quality checks with structured issue reporting.")

    # Get current dataset
    df, file_name = get_current_dataset()

    if df is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset or use the sample dataset first.")
        if st.button("← Go to Upload", use_container_width=True):
            st.switch_page(get_page("Upload Dataset"))
        return

    # Sidebar: Analysis options
    with st.sidebar:
        st.header("Analysis Options")

        outlier_method = st.selectbox(
            "Outlier Detection Method",
            ["iqr", "zscore"],
            index=0,
            help="IQR: Interquartile Range | Z-Score: Standard deviations from mean",
        )

        outlier_multiplier = st.slider(
            "IQR Multiplier",
            1.0, 3.0, 1.5, 0.1,
            help="Multiplier for IQR outlier bounds (default: 1.5)",
        ) if outlier_method == "iqr" else None

        outlier_threshold = st.slider(
            "Z-Score Threshold",
            2.0, 5.0, 3.0, 0.1,
            help="Z-score threshold for outliers (default: 3.0)",
        ) if outlier_method == "zscore" else None

        run_analysis = st.button(
            "🔄 Run Quality Analysis",
            type="primary",
            use_container_width=True,
        )

    # Run analysis if button clicked or not yet run
    if run_analysis or st.session_state.get("quality_report") is None:
        run_quality_analysis(df, outlier_method, outlier_multiplier, outlier_threshold)

    # Display results
    quality_report = st.session_state.get("quality_report")
    if quality_report:
        display_quality_report(quality_report, df)

        # Navigation hint
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("✅ Quality analysis complete. Navigate to **AI Recommendations** for AI-powered preprocessing suggestions.")
        with col2:
            if st.button("Next: AI Recommendations →", type="primary", use_container_width=True):
                st.switch_page(get_page("AI Recommendations"))


def run_quality_analysis(
    df: pd.DataFrame,
    outlier_method: str,
    outlier_multiplier: Optional[float],
    outlier_threshold: Optional[float],
):
    """Run quality analysis with progress indicators."""
    with st.status("Running quality analysis...", expanded=True) as status:
        try:
            # Step 1: Profile dataset
            st.write("📊 Profiling dataset...")
            profile = profile_dataset_cached(df)

            # Step 2: Run quality checks
            st.write("🔍 Running quality checks...")
            quality_report = run_quality_checks_cached(
                df,
                outlier_method=outlier_method,
                outlier_multiplier=outlier_multiplier or 1.5,
                outlier_threshold=outlier_threshold or 3.0,
            )

            # Store in session state
            st.session_state.quality_report = quality_report
            st.session_state.dataset_profile = profile

            status.update(label="✅ Quality analysis complete!", state="complete")

        except Exception as e:
            status.update(label=f"❌ Analysis failed: {e}", state="error")
            st.error(f"Quality analysis failed: {e}")


def display_quality_report(quality_report: dict, df: pd.DataFrame):
    """Display quality report with visualizations."""
    # Overall summary
    st.subheader("📋 Quality Report Summary")

    total_issues = quality_report.get("total_issues", 0)
    overall_summary = quality_report.get("overall_summary", "")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Issues", total_issues)
    with col2:
        # Count by severity
        issues = quality_report.get("issues", [])
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in issues:
            sev = issue.get("severity", "low").lower()
            if sev in severity_counts:
                severity_counts[sev] += 1
        st.metric("Critical", severity_counts["critical"], delta_color="inverse")
    with col3:
        st.metric("High", severity_counts["high"], delta_color="inverse")
    with col4:
        st.metric("Medium + Low", severity_counts["medium"] + severity_counts["low"])

    if overall_summary:
        st.info(overall_summary)

    # Issue categories tabs
    st.subheader("📊 Issues by Category")

    # Define category mapping
    category_map = {
        "missing_value_issues": ("Missing Values", "🔴"),
        "duplicate_issues": ("Duplicates", "🟠"),
        "outlier_issues": ("Outliers", "🟡"),
        "type_issues": ("Data Types", "🔵"),
        "categorical_consistency_issues": ("Categorical Consistency", "🟣"),
        "invalid_value_issues": ("Invalid Values", "🔴"),
        "cardinality_issues": ("Cardinality", "🟠"),
        "constant_column_issues": ("Constant Columns", "🔵"),
        "correlation_issues": ("Correlations", "🟣"),
        "id_detection_issues": ("ID Detection", "🔵"),
        "leakage_issues": ("Data Leakage", "🔴"),
        "string_issues": ("String Issues", "🟡"),
    }

    # Create tabs for categories with issues
    tabs_data = []
    for cat_key, (cat_label, icon) in category_map.items():
        cat_issues = quality_report.get(cat_key, [])
        if cat_issues:
            tabs_data.append((cat_key, cat_label, icon, len(cat_issues)))

    if tabs_data:
        tab_labels = [f"{icon} {label} ({count})" for _, label, icon, count in tabs_data]
        tabs = st.tabs(tab_labels)

        for tab, (cat_key, cat_label, icon, count) in zip(tabs, tabs_data):
            with tab:
                display_category_issues(quality_report.get(cat_key, []))
    else:
        st.success("🎉 No quality issues detected!")

    # All issues table
    with st.expander("📋 All Issues (Detailed Table)", expanded=False):
        display_all_issues_table(quality_report.get("issues", []))

    # Column profiles
    with st.expander("📈 Column Profiles", expanded=False):
        display_column_profiles(quality_report.get("column_info", []))


def display_category_issues(issues: list):
    """Display issues for a specific category."""
    if not issues:
        st.info("No issues in this category.")
        return

    for issue in issues:
        severity = issue.get("severity", "low").lower()
        color = get_severity_color(severity)

        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{issue.get('issue_type', 'Unknown').replace('_', ' ').title()}**")
                if issue.get("column"):
                    st.caption(f"Column: {issue['column']}")
                if issue.get("description"):
                    st.write(issue["description"])
            with col2:
                st.badge(severity.upper(), color=color)

            # Evidence
            evidence = issue.get("evidence", {})
            if evidence:
                with st.expander("📎 Evidence", expanded=False):
                    st.json(evidence)

            # Recommended action
            if issue.get("recommended_action"):
                st.caption(f"💡 Recommended: {issue['recommended_action']}")


def display_all_issues_table(issues: list):
    """Display all issues in a sortable table."""
    if not issues:
        st.info("No issues to display.")
        return

    table_data = []
    for issue in issues:
        table_data.append({
            "Severity": issue.get("severity", "low").upper(),
            "Type": issue.get("issue_type", "").replace("_", " ").title(),
            "Column": issue.get("column") or "—",
            "Count": issue.get("count", 0),
            "Percentage": f"{issue.get('percentage', 0):.1f}%",
            "Description": issue.get("description", "")[:100] + ("..." if len(issue.get("description", "")) > 100 else ""),
            "Recommended Action": issue.get("recommended_action", "") or "—",
        })

    issues_df = pd.DataFrame(table_data)
    st.dataframe(
        issues_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Severity": st.column_config.TextColumn(
                "Severity",
                help="Issue severity level",
            ),
            "Percentage": st.column_config.TextColumn("Percentage"),
        },
    )


def display_column_profiles(column_info: list):
    """Display column profiling information."""
    if not column_info:
        st.info("No column profiles available.")
        return

    profile_data = []
    for col in column_info:
        profile_data.append({
            "Column": col.get("name", ""),
            "Type": col.get("data_type", ""),
            "Column Type": col.get("column_type", ""),
            "Non-Null": col.get("non_null_count", 0),
            "Missing": col.get("missing_count", 0),
            "Missing %": f"{col.get('missing_percentage', 0):.1f}%",
            "Unique": col.get("unique_count", 0),
            "Min": col.get("min_value", "—"),
            "Max": col.get("max_value", "—"),
            "Mean": col.get("mean_value", "—"),
            "Std": col.get("std_value", "—"),
        })

    profile_df = pd.DataFrame(profile_data)
    st.dataframe(profile_df, hide_index=True, width="stretch")
