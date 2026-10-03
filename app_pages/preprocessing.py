"""
Preprocessing Pipeline Page

Applies approved preprocessing actions to the dataset.
Shows real-time progress and allows preview of changes.
"""

import streamlit as st
import pandas as pd
from typing import Optional

from app_pages.prewarming import (
    init_session_state,
    get_current_dataset,
    run_preprocessing_pipeline_cached,
    format_bytes,
)
from app_pages.navigation_registry import get_page


def preprocessing_page():
    """Main preprocessing pipeline page function."""
    init_session_state()

    # Page header
    st.title("⚙️ Preprocessing Pipeline")
    st.caption("Apply approved preprocessing actions to transform your dataset.")

    # Get current dataset and approved actions
    df, file_name = get_current_dataset()
    approved_actions = st.session_state.get("approved_actions", [])
    ai_recommendations = st.session_state.get("ai_recommendations", {})

    if df is None:
        st.warning("⚠️ No dataset loaded. Please go to Upload Dataset first.")
        if st.button("← Go to Upload", use_container_width=True):
            st.switch_page(get_page("Upload Dataset"))
        return

    if not approved_actions:
        st.warning("⚠️ No approved actions. Please go to AI Recommendations to approve actions.")
        if st.button("← Go to AI Recommendations", use_container_width=True):
            st.switch_page(get_page("AI Recommendations"))
        return

    # Sidebar: Pipeline options
    with st.sidebar:
        st.header("Pipeline Actions")

        for i, action in enumerate(approved_actions):
            with st.container(border=True):
                st.write(f"**{i+1}. {action['action'].replace('_', ' ').title()}**")
                if action.get("columns"):
                    st.caption(f"Columns: {', '.join(action['columns'])}")
                if action.get("parameters"):
                    st.caption(f"Params: {action['parameters']}")

        st.divider()

        # Option to add manual actions
        with st.expander("➕ Add Manual Action"):
            add_manual_action_ui()

        run_pipeline = st.button(
            "🚀 Run Pipeline",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("processed_df") is not None,
        )

        if st.session_state.get("processed_df") is not None:
            if st.button("🔄 Reset & Re-run", use_container_width=True):
                reset_pipeline()
                st.rerun()

    # Show approved actions summary
    st.subheader("📋 Approved Actions")
    for i, action in enumerate(approved_actions):
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{i+1}. {action['action'].replace('_', ' ').title()}**")
                if action.get("columns"):
                    st.caption(f"Columns: {', '.join(action['columns'])}")
            with col2:
                if st.button("🗑️", key=f"remove_{i}", help="Remove this action"):
                    st.session_state.approved_actions.pop(i)
                    st.rerun()

    # Run pipeline
    if run_pipeline:
        run_pipeline_with_progress(df, approved_actions)

    # Display results
    processed_df = st.session_state.get("processed_df")
    summary = st.session_state.get("preprocessing_summary")

    if processed_df is not None:
        display_pipeline_results(df, processed_df, summary)

        # Navigation hint
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("✅ Preprocessing complete. Navigate to **Validation & Results** to verify the results.")
        with col2:
            if st.button("Next: Validation & Results →", type="primary", use_container_width=True):
                st.switch_page(get_page("Validation & Results"))


def add_manual_action_ui():
    """UI for adding manual preprocessing actions."""
    from ai.schemas import PreprocessingAction

    action_options = PreprocessingAction.all_actions()
    selected_action = st.selectbox("Action", action_options, key="manual_action_select")

    df, _ = get_current_dataset()
    if df is not None:
        available_columns = list(df.columns)
        selected_columns = st.multiselect("Columns", available_columns, key="manual_action_columns")
    else:
        selected_columns = []

    # Action-specific parameters
    parameters = {}

    if selected_action in ["remove_outliers", "clip_outliers"]:
        parameters["multiplier"] = st.number_input("IQR Multiplier", 1.0, 5.0, 1.5, 0.1, key="manual_multiplier")

    if selected_action == "normalize_categorical_case":
        parameters["case"] = st.selectbox("Case", ["lower", "upper", "title"], key="manual_case")

    if selected_action == "convert_to_numeric":
        parameters["errors"] = st.selectbox("Errors", ["raise", "coerce", "ignore"], key="manual_errors")

    if selected_action == "convert_to_datetime":
        parameters["format"] = st.text_input("Format (optional)", key="manual_format")
        parameters["errors"] = st.selectbox("Errors", ["raise", "coerce", "ignore"], key="manual_dt_errors")

    if st.button("Add Action", key="add_manual_action"):
        if selected_columns or selected_action in ["drop_duplicates"]:
            new_action = {
                "action": selected_action,
                "columns": selected_columns,
                "parameters": parameters,
            }
            st.session_state.approved_actions.append(new_action)
            st.success(f"Added: {selected_action}")
            st.rerun()
        else:
            st.error("Please select at least one column")


def run_pipeline_with_progress(df: pd.DataFrame, actions: list):
    """Run preprocessing pipeline with progress indicators."""
    with st.status("Running preprocessing pipeline...", expanded=True) as status:
        try:
            st.write(f"🔄 Applying {len(actions)} preprocessing actions...")

            result = run_preprocessing_pipeline_cached(df, actions)

            processed_df = result["processed_df"]
            summary = result["summary"]

            st.session_state.processed_df = processed_df
            st.session_state.preprocessing_summary = summary

            status.update(label="✅ Pipeline executed successfully!", state="complete")

        except Exception as e:
            status.update(label=f"❌ Pipeline failed: {e}", state="error")
            st.error(f"Preprocessing pipeline failed: {e}")


def display_pipeline_results(original_df: pd.DataFrame, processed_df: pd.DataFrame, summary: dict):
    """Display preprocessing pipeline results."""
    st.subheader("📊 Pipeline Results")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Actions Applied", summary.get("actions_applied", 0))
    with col2:
        st.metric("Rows Before", f"{summary.get('rows_before', 0):,}")
    with col3:
        st.metric("Rows After", f"{summary.get('rows_after', 0):,}")
    with col4:
        rows_removed = summary.get("rows_removed", 0)
        st.metric("Rows Removed", f"{rows_removed:,}", delta_color="inverse" if rows_removed > 0 else "off")

    # Action logs
    if summary.get("action_logs"):
        with st.expander("📝 Action Logs", expanded=True):
            for i, log in enumerate(summary["action_logs"]):
                with st.container(border=True):
                    st.write(f"**Action {i+1}: {log.get('action', 'Unknown')}**")
                    if log.get("columns"):
                        st.caption(f"Columns: {', '.join(log['columns'])}")
                    if log.get("rows_affected"):
                        st.caption(f"Rows affected: {log['rows_affected']}")
                    if log.get("details"):
                        st.json(log["details"])

    # Side-by-side comparison
    st.subheader("🔄 Before vs After Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Dataset**")
        st.dataframe(original_df.head(20), hide_index=True, width="stretch")
        st.caption(f"{len(original_df)} rows × {len(original_df.columns)} columns")

    with col2:
        st.write("**Processed Dataset**")
        st.dataframe(processed_df.head(20), hide_index=True, width="stretch")
        st.caption(f"{len(processed_df)} rows × {len(processed_df.columns)} columns")

    # Column changes
    st.subheader("📈 Column Changes")
    col_changes = []
    all_columns = set(original_df.columns) | set(processed_df.columns)
    for col in sorted(all_columns):
        in_original = col in original_df.columns
        in_processed = col in processed_df.columns

        if in_original and in_processed:
            orig_dtype = str(original_df[col].dtype)
            proc_dtype = str(processed_df[col].dtype)
            dtype_changed = orig_dtype != proc_dtype
            col_changes.append({
                "Column": col,
                "Status": "✅ Kept" if not dtype_changed else "🔄 Type Changed",
                "Original Type": orig_dtype,
                "Processed Type": proc_dtype,
            })
        elif in_original and not in_processed:
            col_changes.append({
                "Column": col,
                "Status": "🗑️ Removed",
                "Original Type": str(original_df[col].dtype),
                "Processed Type": "—",
            })
        elif not in_original and in_processed:
            col_changes.append({
                "Column": col,
                "Status": "➕ Added",
                "Original Type": "—",
                "Processed Type": str(processed_df[col].dtype),
            })

    if col_changes:
        st.dataframe(pd.DataFrame(col_changes), hide_index=True, width="stretch")


def reset_pipeline():
    """Reset pipeline results."""
    keys_to_reset = [
        "processed_df",
        "preprocessing_summary",
        "validation_report",
        "validation_config",
        "show_validation_download",
        "show_full_report_download",
        "last_full_report",
        "last_export_name",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
