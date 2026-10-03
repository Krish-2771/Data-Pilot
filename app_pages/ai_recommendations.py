"""
AI Recommendations Page

Generates AI-powered preprocessing recommendations using NVIDIA NIM.
Displays recommendations with confidence scores and approval workflow.
"""

import streamlit as st
import pandas as pd
from typing import Optional, List

from app_pages.prewarming import (
    init_session_state,
    get_current_dataset,
    generate_ai_recommendations_cached,
    get_severity_color,
)
from app_pages.navigation_registry import get_page


def ai_recommendations_page():
    """Main AI recommendations page function."""
    init_session_state()

    # Page header
    st.title("🤖 AI Recommendations")
    st.caption("AI-powered preprocessing recommendations using NVIDIA NIM (Nemotron 3.5).")

    # Get current dataset and quality report
    df, file_name = get_current_dataset()
    quality_report = st.session_state.get("quality_report")

    if df is None:
        st.warning("⚠️ No dataset loaded. Please go to Upload Dataset first.")
        if st.button("← Go to Upload", use_container_width=True):
            st.switch_page(get_page("Upload Dataset"))
        return

    if quality_report is None:
        st.warning("⚠️ No quality report available. Please run Quality Analysis first.")
        if st.button("← Go to Quality Analysis", use_container_width=True):
            st.switch_page(get_page("Quality Analysis"))
        return

    # Sidebar: AI Configuration
    with st.sidebar:
        st.header("AI Configuration")

        # Check NIM client
        try:
            from ai.client import NIMClient
            from ai.exceptions import MissingAPIKeyError
            client = NIMClient.from_env()
            st.success(f"✅ Connected to {client.model}")
            st.caption(f"Temperature: {client.temperature} | Max Tokens: {client.max_tokens}")
        except MissingAPIKeyError:
            st.error("❌ NVIDIA_API_KEY not configured")
            st.info("Set NVIDIA_API_KEY in your environment or Streamlit Cloud Secrets to enable AI recommendations")
            return
        except Exception as e:
            st.error(f"❌ NIM client error: {e}")
            return

        use_compact_prompt = st.checkbox(
            "Use compact prompt (faster)",
            value=True,
            help="Uses shorter prompt for faster inference",
        )

        generate_btn = st.button(
            "🚀 Generate Recommendations",
            type="primary",
            use_container_width=True,
        )

    # Generate recommendations
    if generate_btn or (st.session_state.get("ai_recommendations") is None and not st.session_state.get("ai_rec_failed", False)):
        generate_recommendations(quality_report, use_compact_prompt)

    # Display recommendations
    ai_response = st.session_state.get("ai_recommendations")
    if ai_response:
        display_ai_recommendations(ai_response, df)

        # Navigation hint
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("✅ AI recommendations generated. Navigate to **Preprocessing Pipeline** to approve and apply actions.")
        with col2:
            if st.button("Next: Preprocessing Pipeline →", type="primary", use_container_width=True):
                st.switch_page(get_page("Preprocessing Pipeline"))


def generate_recommendations(quality_report: dict, use_compact_prompt: bool = True):
    """Generate AI recommendations with progress."""
    st.session_state.ai_rec_failed = False
    with st.status("Generating AI recommendations...", expanded=True) as status:
        try:
            st.write("🧠 Sending quality report to NVIDIA NIM...")
            st.write("⏳ Waiting for model response...")

            result = generate_ai_recommendations_cached(
                quality_report, use_compact_prompt=use_compact_prompt
            )

            if "error" in result:
                status.update(label=f"❌ Failed: {result['error']}", state="error")
                st.error(f"AI recommendation failed: {result['error']}")
                st.session_state.ai_rec_failed = True
                return

            st.session_state.ai_recommendations = result
            st.session_state.ai_rec_failed = False
            status.update(label="✅ Recommendations generated!", state="complete")

        except Exception as e:
            status.update(label=f"❌ Failed: {e}", state="error")
            st.error(f"Failed to generate recommendations: {e}")
            st.session_state.ai_rec_failed = True


def display_ai_recommendations(ai_response: dict, df: pd.DataFrame):
    """Display AI recommendations with approval workflow."""
    # Summary
    st.subheader("📋 AI Analysis Summary")
    st.write(ai_response.get("summary", "No summary provided."))

    if ai_response.get("reasoning"):
        with st.expander("🧠 AI Reasoning", expanded=False):
            st.write(ai_response["reasoning"])

    recommendations = ai_response.get("recommendations", [])

    if not recommendations:
        st.success("🎉 No preprocessing actions recommended - dataset looks clean!")
        return

    st.subheader(f"🎯 Recommended Actions ({len(recommendations)})")

    # Initialize approved actions if not present
    if "approved_actions" not in st.session_state:
        st.session_state.approved_actions = []

    # Queue recommendations that explicitly do not require user approval.
    for rec in recommendations:
        if rec.get("requires_approval", True):
            continue
        action_data = {
            "action": rec["action"],
            "columns": rec.get("columns", []),
            "parameters": rec.get("parameters", {}),
        }
        if action_data not in st.session_state.approved_actions:
            st.session_state.approved_actions.append(action_data)

    # Display each recommendation as a card
    for i, rec in enumerate(recommendations):
        display_recommendation_card(i, rec, df)

    # Bulk actions (outside the loop to avoid duplicate keys)
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Approve All", key="approve_all", use_container_width=True):
            approve_all_actions(recommendations)
    with col2:
        if st.button("❌ Reject All", key="reject_all", use_container_width=True):
            reject_all_actions(recommendations)
    with col3:
        approved_count = len(st.session_state.approved_actions)
        st.metric("Approved", f"{approved_count}/{len(recommendations)}")


def display_recommendation_card(index: int, rec: dict, df: pd.DataFrame):
    """Display a single recommendation card with approval controls."""
    action = rec.get("action", "")
    columns = rec.get("columns", [])
    reason = rec.get("reason", "")
    confidence = rec.get("confidence", 0.0)
    requires_approval = rec.get("requires_approval", True)
    parameters = rec.get("parameters", {})

    # Check if already approved
    is_approved = any(
        a.get("action") == action and a.get("columns") == columns
        for a in st.session_state.approved_actions
    )

    # Confidence color
    conf_color = "green" if confidence >= 0.8 else "orange" if confidence >= 0.5 else "red"

    with st.container(border=True):
        # Header row
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**{action.replace('_', ' ').title()}**")
            if columns:
                st.caption(f"Columns: {', '.join(columns)}")
        with col2:
            st.metric("Confidence", f"{confidence:.0%}", delta_color="off")
        with col3:
            if is_approved:
                st.badge("✅ APPROVED", color="green")
            elif requires_approval:
                st.badge("⏳ PENDING", color="orange")
            else:
                st.badge("AUTO", color="blue")

        # Reason
        st.write(reason)

        # Parameters
        if parameters:
            with st.expander("⚙️ Parameters", expanded=False):
                st.json(parameters)

        # Approval controls
        if requires_approval and not is_approved:
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("✅ Approve", key=f"approve_{index}", type="primary"):
                    approve_action(index, rec)
            with col2:
                if st.button("❌ Reject", key=f"reject_{index}"):
                    reject_action(index, rec)
        elif is_approved:
            st.caption("✅ This action has been approved and will be applied in the preprocessing pipeline.")
        else:
            st.caption("ℹ️ This action does not require approval and will be applied automatically.")


def approve_action(index: int, rec: dict):
    """Approve a single recommendation."""
    action_data = {
        "action": rec["action"],
        "columns": rec.get("columns", []),
        "parameters": rec.get("parameters", {}),
    }
    if action_data not in st.session_state.approved_actions:
        st.session_state.approved_actions.append(action_data)
    st.success(f"Approved: {rec['action']}")
    st.rerun()


def reject_action(index: int, rec: dict):
    """Reject a single recommendation (mark as not approved)."""
    # Just remove if it was somehow added
    st.session_state.approved_actions = [
        a for a in st.session_state.approved_actions
        if not (a.get("action") == rec["action"] and a.get("columns") == rec.get("columns", []))
    ]
    st.info(f"Rejected: {rec['action']}")
    st.rerun()


def approve_all_actions(recommendations: list):
    """Approve all recommendations that require approval."""
    for rec in recommendations:
        if rec.get("requires_approval", True):
            action_data = {
                "action": rec["action"],
                "columns": rec.get("columns", []),
                "parameters": rec.get("parameters", {}),
            }
            if action_data not in st.session_state.approved_actions:
                st.session_state.approved_actions.append(action_data)
    st.success("All recommendations approved!")
    st.rerun()


def reject_all_actions(recommendations: list):
    """Reject all recommendations."""
    st.session_state.approved_actions = []
    st.info("All recommendations rejected.")
    st.rerun()
