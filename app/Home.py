"""Explicit page registry for public and full platform profiles."""
from app._bootstrap import bootstrap

bootstrap()

import streamlit as st

from backend.config import SETTINGS


st.set_page_config(
    page_title="Explainable RAG Studio", page_icon="◈", layout="wide",
    initial_sidebar_state="expanded",
)

public_pages = [
    st.Page("public_home.py", title="Home", icon=":material/home:", default=True),
    st.Page("pages/1_What_is_RAG.py", title="What is RAG?", icon=":material/school:"),
    st.Page("pages/3_Ask_and_Explain.py", title="Ask & Explain", icon=":material/search:"),
    st.Page("pages/9_Results.py", title="Results", icon=":material/analytics:"),
]

if SETTINGS.low_memory_demo:
    navigation = st.navigation(public_pages)
else:
    navigation = st.navigation({
        "Showcase": public_pages,
        "Workspace": [
            st.Page("pages/2_Ingest_and_Index.py", title="Ingest & Index"),
            st.Page("pages/4_Embedding_Explorer.py", title="Embedding Explorer"),
            st.Page("pages/5_Evaluation.py", title="Evaluation"),
            st.Page("pages/6_Latency_Dashboard.py", title="Latency Dashboard"),
            st.Page("pages/7_Grounding_Review.py", title="Grounding Review"),
            st.Page("pages/8_Security_Center.py", title="Security Center"),
        ],
    })

navigation.run()
