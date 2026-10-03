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
    """Render a compact footer at the bottom of every application page."""
    st.divider()
    brand, link = st.columns([4, 1])
    with brand:
        st.markdown("**DATA-PILOT** · AI-powered data quality assistant")
        st.caption("Analyze · Clean · Validate · Prepare")
    with link:
        st.page_link(
            get_page("Footer"),
            label="About Data-Pilot & Developers",
            icon=":material/arrow_forward:",
        )


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
            ("🔎", "Analyze", "Profile your dataset and find quality issues."),
            ("🧹", "Clean", "Apply preprocessing actions to your data."),
            ("✅", "Validate", "Review checks after processing."),
            ("📦", "Prepare", "Export the cleaned dataset and report."),
        ],
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {icon} {title}")
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
    st.caption("© 2026 Data-Pilot · Making datasets reliable, consistent, and ML-ready.")

    st.markdown("### Your custom content")
    st.info("Add your own sections, links, or other Streamlit elements below.")
    # Add any custom Streamlit content below this line.
