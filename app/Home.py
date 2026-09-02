import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import streamlit as st
from backend.config import SETTINGS
from app.security_ui import security_context, tenant_store
from app.ui import configure_page, feature_card, footer, metric_card, page_header, section, status_pills

configure_page("Overview")
context = security_context(allow_anonymous=True)
store = tenant_store(context)
index_ready = store.index is not None
items = store.meta.get("items", []) if index_ready else []

page_header("AI reliability workbench", "See exactly why your RAG system answered that way.", "Build a document index, inspect every retrieval decision, validate grounded answers, and measure performance from one focused workspace.")
status_pills([("Vector index ready" if index_ready else "Index not built", index_ready), ("Gemini connected" if SETTINGS.gemini_api_key.strip() else "Extractive mode", bool(SETTINGS.gemini_api_key.strip())), (SETTINGS.embedding_model.split("/")[-1], True)])

cols = st.columns(4)
values = [("Documents", str(len({x.get('source') for x in items})), "Indexed source files"), ("Knowledge units", f"{len(items):,}", "Searchable chunks"), ("Retriever", "Hybrid", "Dense + lexical + reranking"), ("Grounding", "Strict", "Unsupported claims removed")]
for col, value in zip(cols, values):
    with col: metric_card(*value)

section("A complete retrieval workflow", "Move from raw documents to evidence-backed answers without losing visibility into the pipeline.")
for col, card in zip(st.columns(3), [("01 · INGEST", "Build the knowledge layer", "Parse PDFs, DOCX, HTML, Markdown, tables, and scans into versioned contextual chunks with durable lifecycle jobs."), ("02 · VERIFY", "Ask and inspect", "Trace each atomic claim to exact evidence, local entailment scores, conflicts, and strict filtering decisions."), ("03 · EVALUATE", "Measure the system", "Run repeatable retrieval and grounding benchmarks, surface failures, and monitor end-to-end latency.")]):
    with col: feature_card(*card)

section("Start here", "Follow the shortest path to a working, explainable query.")
left, right = st.columns([1.3, 1])
with left:
    st.markdown("#### New workspace")
    st.write("Index a small, representative document set and tune chunking before you query it.")
    if st.button("Open ingestion workspace", type="primary", use_container_width=True): st.switch_page("pages/2_Ingest_and_Index.py")
with right:
    st.markdown("#### Index already built")
    st.write("Go straight to retrieval and inspect the evidence behind the response.")
    if st.button("Ask a grounded question", disabled=not index_ready, use_container_width=True): st.switch_page("pages/3_Ask_and_Explain.py")
footer()
