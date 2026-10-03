"""
Data-Pilot Streamlit Application
Main entry point with multi-page navigation and pre-warming strategy.
"""

import streamlit as st
from openai import OpenAI

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Data-Pilot: AI Dataset Health Analyzer",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for pre-warming
if "prewarm_ready" not in st.session_state:
    st.session_state.prewarm_ready = False

if "cached_sample_df" not in st.session_state:
    st.session_state.cached_sample_df = None

# Navigation pages
from app_pages.upload import upload_page
from app_pages.quality_analysis import quality_analysis_page
from app_pages.ai_recommendations import ai_recommendations_page
from app_pages.preprocessing import preprocessing_page
from app_pages.validation import validation_page
from app_pages.footer import footer_page, render_footer
from app_pages.navigation_registry import register_pages

# Create page objects (flat list for st.navigation)
pages = [
    st.Page(upload_page, title="Upload Dataset", icon=":material/upload_file:"),
    st.Page(quality_analysis_page, title="Quality Analysis", icon=":material/analytics:"),
    st.Page(ai_recommendations_page, title="AI Recommendations", icon=":material/psychology:"),
    st.Page(preprocessing_page, title="Preprocessing Pipeline", icon=":material/build:"),
    st.Page(validation_page, title="Validation & Results", icon=":material/check_circle:"),
    st.Page(footer_page, title="Footer", icon=":material/notes:"),
]

# Navigation
register_pages(pages)
pg = st.navigation(pages, position="sidebar")
pg.run()
if pg.title != "Footer":
    render_footer()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=st.secrets["NVIDIA_API_KEY"])