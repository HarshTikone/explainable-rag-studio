import streamlit as st
from app._bootstrap import bootstrap
bootstrap()
from app.ui import configure_page, feature_card, footer, page_header, section
from backend.config import SETTINGS

configure_page("How RAG works", "◎")
page_header("RAG fundamentals", "From a question to a grounded answer.", "Understand the retrieval pipeline, where hallucinations enter, and how source evidence makes answers verifiable.")

retrieval_copy = (
    "BM25 matches the question's terms against the sanitized corpus and ranks the strongest passages."
    if SETTINGS.low_memory_demo else
    "Dense and lexical retrieval rank the strongest supporting passages, with optional reranking."
)
for col, card in zip(st.columns(3), [("01 · MODEL", "Fluent is not factual", "An LLM predicts useful text, but it does not automatically know your private or newly uploaded documents."), ("02 · RETRIEVAL", "Find the evidence first", retrieval_copy), ("03 · GENERATION", "Answer from context", "Only selected evidence is sent to the model, with instructions to abstain when support is insufficient.")]):
    with col: feature_card(*card)

section("Pipeline anatomy", "Each stage is visible in this studio so quality problems can be traced to their source.")
if SETTINGS.low_memory_demo:
    st.markdown("""
1. **Prepare** — load the committed, sanitized portfolio corpus.
2. **Chunk** — split each source into bounded, traceable knowledge units.
3. **Retrieve** — use BM25 to rank chunks by lexical relevance.
4. **Select** — let Gemini choose numbered evidence sentences, or use the deterministic local fallback.
5. **Verify** — require every displayed claim to be exact text from its cited chunk.
6. **Explain** — expose citations, ranked evidence, latency, and the generation mode.
""")
else:
    st.markdown("""
1. **Parse** — extract page-level text from the source document.
2. **Chunk** — split text into overlapping, token-bounded knowledge units.
3. **Embed** — convert each chunk into a semantic vector.
4. **Retrieve** — combine dense and lexical retrieval with optional reranking.
5. **Generate** — answer using only the selected context.
6. **Verify** — expose citations, retrieval scores, latency, and benchmark results.
""")

with st.container(border=True):
    st.markdown("#### Why explainability matters")
    st.write("A plausible answer can still be wrong. Inspecting the retrieved evidence tells you whether the failure came from the knowledge base, chunking, ranking, or generation.")
    if SETTINGS.low_memory_demo:
        left, right = st.columns(2)
        if left.button("Try a tested question", type="primary", use_container_width=True):
            st.switch_page("pages/3_Ask_and_Explain.py")
        if right.button("Review measured results", use_container_width=True):
            st.switch_page("pages/9_Results.py")
    elif st.button("Build your first index", type="primary"):
        st.switch_page("pages/2_Ingest_and_Index.py")
footer()
