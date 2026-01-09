import streamlit as st
import pandas as pd
import time
import json

from backend.config import SETTINGS
from backend.embeddings import Embedder
from backend.vectorstore import FaissStore
from backend.retriever import retrieve
from backend.qa import answer_with_optional_llm
from backend.telemetry import log_run
from backend.utils import now_ms
from app._bootstrap import bootstrap
bootstrap()

st.title("Ask & Explain (Retrieval + Citations)")

# Load index
store = FaissStore(SETTINGS.index_dir)
loaded = store.load()
if not loaded:
    st.warning("No index found. Go to “Ingest & Index” first.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.slider("Top-K chunks", 3, 12, SETTINGS.top_k, step=1)
with col2:
    use_mmr = st.toggle("Use MMR (diversity)", value=SETTINGS.use_mmr)
with col3:
    embed_model = st.text_input("Embedding model", SETTINGS.embedding_model)

question = st.text_area("Your question", height=120, placeholder="Ask something from your documents...")

use_gemini = True  # We default to Gemini; fallback occurs if key missing
st.caption("Generator: Gemini (falls back to extractive if no API key detected)")

gemini_client = None
gemini_model = SETTINGS.gemini_model

# Init Gemini client
# If GEMINI_API_KEY is set in env, genai.Client() will pick it up automatically.
# You can also pass api_key explicitly.
from google import genai

gemini_client = None
use_gemini = False

if SETTINGS.gemini_api_key.strip():
    try:
        gemini_client = genai.Client(api_key=SETTINGS.gemini_api_key)
        use_gemini = True
    except Exception as e:
        st.warning("Gemini API key detected but client initialization failed.")
        use_gemini = False
else:
    st.info("No Gemini API key found — using extractive fallback.")


# If no key is actually available, requests will fail; we detect that at runtime and fallback.
if st.button("Ask", type="primary", disabled=not question.strip()):
    embedder = Embedder(embed_model)

    t0 = time.time()
    retrieved = retrieve(
        store=store,
        embed_query_fn=embedder.embed_query,
        query=question,
        top_k=top_k,
        use_mmr=use_mmr
    )
    t1 = time.time()

    # Try Gemini, fallback if it errors (missing key / quota / etc.)
    try:
        out = answer_with_optional_llm(
                    question=question,
                    retrieved_items=retrieved,
                    use_gemini=use_gemini,
                    gemini_client=gemini_client,
                    gemini_model=gemini_model
                )
    except Exception:
        out = answer_with_optional_llm(
            question=question,
            retrieved_items=retrieved,
            use_gemini=False,
            gemini_client=None,
            gemini_model=gemini_model
        )

    t2 = time.time()

    retrieval_ms = int((t1 - t0) * 1000)
    generation_ms = int((t2 - t1) * 1000)
    total_ms = int((t2 - t0) * 1000)

    st.subheader("Answer")
    st.write(out["answer"])

    st.subheader("Citations (2–3)")
    if out["citations"]:
        st.table(pd.DataFrame(out["citations"]))
    else:
        st.write("No citations available.")

    st.subheader("Retrieved chunks (explainability)")
    rows = []
    for score, item in sorted(retrieved, key=lambda x: x[0], reverse=True):
        rows.append({
            "score": round(score, 4),
            "chunk_id": item["chunk_id"],
            "source": item["source"],
            "page": item["page"],
            "preview": item["text"][:220].replace("\n", " ") + "..."
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Context sent to the generator")
    st.code(out["context"][:6000])

    st.subheader("Latency")
    st.write({
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "total_ms": total_ms
    })

    # Log run to SQLite
    log_run({
        "ts_ms": now_ms(),
        "query": question,
        "top_k": top_k,
        "use_mmr": use_mmr,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "total_ms": total_ms,
        "citations": json.dumps(out["citations"], ensure_ascii=False)
    })
