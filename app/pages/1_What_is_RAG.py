import streamlit as st
from app._bootstrap import bootstrap
bootstrap()
from app.ui import configure_page, feature_card, footer, page_header, section

configure_page("How RAG works", "◎")
page_header("RAG fundamentals", "From a question to a grounded answer.", "Understand the retrieval pipeline, where hallucinations enter, and how source evidence makes answers verifiable.")

for col, card in zip(st.columns(3), [("01 · MODEL", "Fluent is not factual", "An LLM predicts useful text, but it does not automatically know your private or newly uploaded documents."), ("02 · RETRIEVAL", "Find the evidence first", "The question is embedded and matched against document chunks to locate the strongest supporting passages."), ("03 · GENERATION", "Answer from context", "Only selected evidence is sent to the model, with instructions to abstain when support is insufficient.")]):
    with col: feature_card(*card)

section("Pipeline anatomy", "Each stage is visible in this studio so quality problems can be traced to their source.")
st.markdown("""
1. **Parse** — extract page-level text from the source document.
2. **Chunk** — split text into overlapping, token-bounded knowledge units.
3. **Embed** — convert each chunk into a semantic vector.
4. **Retrieve** — rank the closest chunks for a question using FAISS and optional MMR.
5. **Generate** — answer using only the selected context.
6. **Verify** — expose citations, retrieval scores, latency, and benchmark results.
""")

with st.container(border=True):
    st.markdown("#### Why explainability matters")
    st.write("A plausible answer can still be wrong. Inspecting the retrieved evidence tells you whether the failure came from the knowledge base, chunking, ranking, or generation.")
    if st.button("Build your first index", type="primary"): st.switch_page("pages/2_Ingest_and_Index.py")
footer()
