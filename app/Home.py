import streamlit as st
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
st.set_page_config(page_title="Explainable RAG Studio", layout="wide")

st.title("Explainable RAG Studio")
st.write("""
This app demonstrates **Retrieval-Augmented Generation (RAG)** end-to-end:
- PDF ingestion → chunking (300–500 tokens)
- embeddings → FAISS vector search
- explainable retrieval (scores + chunk viewer)
- answers with citations
- evaluation + latency dashboard

Use the pages in the sidebar to walk through the system like a product demo.
""")

st.info("Tip: Start with “What is RAG?” then go to “Ingest & Index”, then “Ask & Explain”.")
