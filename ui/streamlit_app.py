"""Streamlit UI. T10, T15, T25 (Aasritha) build the topic, outline, report, analysis and chat screens here.

Run:  uv run streamlit run ui/streamlit_app.py
"""

import sys
from pathlib import Path

import httpx
import streamlit as st

# Streamlit puts only this file's folder on the import path; add the project root so `app` can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

st.set_page_config(page_title="ResearchPilot", layout="wide")
st.title("ResearchPilot")
st.caption("Research reports with citations, human review, analysis and a chatbot.")

try:
    response = httpx.get(f"{settings.api_base_url}/health", timeout=5)
    response.raise_for_status()
    st.success(f"API is running (LLM provider: {response.json()['llm_provider']})")
except httpx.HTTPError:
    st.error(f"API is not reachable at {settings.api_base_url}. Start it: uv run uvicorn app.api.main:app --reload")
