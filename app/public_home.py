import streamlit as st

from backend.config import SETTINGS
from app.security_ui import security_context, tenant_store
from app.ui import configure_page, feature_card, footer, metric_card, page_header, section, status_pills


configure_page("Overview")
context = security_context(allow_anonymous=True)
store = tenant_store(context)
index_ready = store.index is not None
items = store.meta.get("items", []) if index_ready else []

page_header(
    "AI reliability workbench",
    "See exactly why your RAG system answered that way.",
    "Ask a tested question, inspect ranked evidence and citations, and review the measured quality of the full platform.",
)
if SETTINGS.low_memory_demo:
    mode_label = "BM25 + Gemini-assisted exact evidence" if SETTINGS.gemini_api_key.strip() else "BM25 + exact extractive fallback"
else:
    mode_label = "Gemini connected" if SETTINGS.gemini_api_key.strip() else "Extractive mode"
retriever_label = "BM25" if SETTINGS.low_memory_demo else SETTINGS.embedding_model.split("/")[-1]
status_pills([
    ("Demo corpus ready" if index_ready else "Index not built", index_ready),
    (mode_label, True),
    (retriever_label, True),
])

cols = st.columns(4)
retriever = ("Lexical", "BM25 on the 512 MB hosted demo") if SETTINGS.low_memory_demo else ("Hybrid", "Dense + lexical + reranking")
grounding = ("Exact evidence", "Verbatim cited answers") if SETTINGS.low_memory_demo else ("Strict", "Unsupported claims removed")
values = [
    ("Documents", str(len({x.get("source") for x in items})), "Sanitized source files"),
    ("Knowledge units", f"{len(items):,}", "Searchable chunks"),
    ("Retriever", *retriever),
    ("Grounding", *grounding),
]
for col, value in zip(cols, values):
    with col:
        metric_card(*value)

section("What this demo proves", "A focused visitor path backed by the repository's larger reliability workbench.")
cards = [
    ("01 · ASK", "Use a tested corpus", "Try exact-term, identifier, hard-negative, and unanswerable questions without uploading private data."),
    ("02 · VERIFY", "Inspect the evidence", "Trace every displayed answer sentence to a retrieved chunk and see what strict verification removed."),
    ("03 · MEASURE", "Read the retained results", "Review retrieval, grounding, runtime, security, benchmark, and known-limitation evidence."),
]
for col, card in zip(st.columns(3), cards):
    with col:
        feature_card(*card)

section("Start here", "The free Render instance may need up to about a minute to wake after inactivity.")
left, right = st.columns(2)
with left:
    st.markdown("#### Ask a grounded question")
    st.write("Choose a sample prompt and inspect the answer, citations, retrieval trace, and fallback mode.")
    if st.button("Open Ask & Explain", type="primary", disabled=not index_ready, use_container_width=True):
        st.switch_page("pages/3_Ask_and_Explain.py")
with right:
    st.markdown("#### Review measured evidence")
    st.write("See which release gates passed, which remain rejected, and how the hosted profile differs from the full platform.")
    if st.button("Open Results", use_container_width=True):
        st.switch_page("pages/9_Results.py")

if SETTINGS.low_memory_demo:
    st.caption("Privacy: the public demo has no upload path. Questions may be sent with sanitized evidence excerpts to Free-tier Gemini; do not enter private information. Raw questions are not retained.")
footer()
