"""Shared site footer and its full details page."""

import streamlit as st

from app_pages.navigation_registry import get_page


TEAM = [
    {
        "name": "Dhruv Pophale",
        "github": "https://github.com/dhruvpophale20",
        "linkedin": "https://www.linkedin.com/in/dhruvpophale20",
    },
    {
        "name": "Krish Kothari",
        "github": "https://github.com/Krish-2771",
        "linkedin": "https://www.linkedin.com/in/krish-kothari-5ba914312",
    },
]


def render_footer() -> None:
    """Render a compact, visually distinct footer at the bottom of each page."""
    st.divider()
    st.markdown(
        """
        <style>
        .st-key-app-footer {
            background: linear-gradient(120deg, #172554, #0f766e);
            border: 1px solid rgba(148, 163, 184, 0.45);
            border-radius: 16px;
            padding: 0.8rem 1.25rem 0.25rem;
            margin-bottom: 1rem;
        }
        .st-key-app-footer [data-testid="stMarkdownContainer"],
        .st-key-app-footer [data-testid="stCaptionContainer"] {
            color: #f8fafc;
        }
        .st-key-app-footer .stButton > button {
            min-height: 3rem;
            border-radius: 10px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="app-footer"):
        brand, action = st.columns([3, 1])
        with brand:
            st.markdown("### DATA-PILOT")
            st.caption("AI-powered data quality assistant | Analyze | Clean | Validate | Prepare")
        with action:
            if st.button(
                "Open Footer Page",
                key="open_footer_page",
                type="primary",
                icon=":material/arrow_forward:",
                use_container_width=True,
            ):
                st.switch_page(get_page("Footer"))


def footer_page() -> None:
    """Render the full, editable Footer page using native Streamlit widgets."""
    st.title("Data-Pilot")
    st.subheader("AI-powered data quality assistant")
    st.write(
        "A workspace for understanding dataset quality, applying reviewed "
        "preprocessing, and validating the result."
    )

    st.markdown("### What you can do")
    features = st.columns(4)
    for column, (icon, title, detail) in zip(
        features,
        [
            ("Search", "Analyze", "Profile your dataset and find quality issues."),
            ("Cleaning Services", "Clean", "Apply preprocessing actions to your data."),
            ("Check Circle", "Validate", "Review checks after processing."),
            ("Inventory 2", "Prepare", "Export the cleaned dataset and report."),
        ],
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"#### :material/{icon.lower().replace(' ', '_')}: {title}")
                st.caption(detail)

    st.markdown("### Project team")
    team_columns = st.columns(len(TEAM))
    for column, member in zip(team_columns, TEAM):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {member['name']}")
                github, linkedin = st.columns(2)
                with github:
                    st.link_button("GitHub", member["github"], use_container_width=True)
                with linkedin:
                    st.link_button("LinkedIn", member["linkedin"], use_container_width=True)

    st.divider()
    st.caption("Copyright 2026 Data-Pilot | Making datasets reliable, consistent, and ML-ready.")

    st.markdown("### Your custom content")
    st.info("Add your own sections, links, or other Streamlit elements below.")
    # Add any custom Streamlit content below this line.
