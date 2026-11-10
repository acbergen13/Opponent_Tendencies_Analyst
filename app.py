"""Streamlit entrypoint.

Stage 0 only renders a placeholder. Selection, refresh, report rendering, and
follow-up chat arrive in Stage 10.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Opponent Tendencies Analyst", layout="wide")
st.title("Opponent Tendencies Analyst")
st.info(
    "Project skeleton only. Data ingestion, analysis, and the report workflow "
    "are under construction (see PLAN.md).",
    icon="🏗️",
)
