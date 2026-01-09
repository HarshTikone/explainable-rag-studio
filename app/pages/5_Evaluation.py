import json
import streamlit as st
import pandas as pd

from google import genai

from backend.config import SETTINGS
from backend.vectorstore import FaissStore
from backend.embeddings import Embedder
from backend.retriever import retrieve
from backend.qa import answer_with_optional_llm
from backend.eval import run_eval

st.title("Evaluation (Accuracy + Failure Analysis)")

# Load FAISS index
store = FaissStore(SETTINGS.index_dir)
if not store.load():
    st.warning("No index found. Build one first (Ingest & Index).")
    st.stop()

st.write(
    """
Upload an eval set JSON in this format:
[
  {"question": "...", "expected": "keyword or phrase"},
  ...
]

Note: This page uses a simple, transparent scoring rule:
- Score = 1 if the expected phrase appears in the model answer (case-insensitive), else 0.
"""
)

eval_file = st.file_uploader("Upload eval_set.json", type=["json"])

col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.slider("Top-K", 3, 12, SETTINGS.top_k, 1)
with col2:
    use_mmr = st.toggle("Use MMR", value=SETTINGS.use_mmr)
with col3:
    embed_model = st.text_input("Embedding model", SETTINGS.embedding_model)

# ----------------------------
# Gemini client init (safe)
# ----------------------------
use_gemini = False
gemini_client = None
gemini_model = SETTINGS.gemini_model

if SETTINGS.gemini_api_key.strip():
    try:
        gemini_client = genai.Client(api_key=SETTINGS.gemini_api_key)
        use_gemini = True
        st.caption(f"Generator: Gemini enabled ({gemini_model})")
    except Exception:
        use_gemini = False
        st.caption("Generator: Gemini key found but init failed → using extractive fallback")
else:
    st.caption("Generator: No Gemini API key found → using extractive fallback")

# ----------------------------
# Run evaluation
# ----------------------------
if st.button("Run Evaluation", type="primary", disabled=not eval_file):
    try:
        eval_items = json.loads(eval_file.getvalue().decode("utf-8"))
        if not isinstance(eval_items, list) or len(eval_items) == 0:
            st.error("Invalid eval_set.json. It must be a non-empty list of objects.")
            st.stop()
    except Exception:
        st.error("Could not parse eval_set.json. Make sure it is valid JSON.")
        st.stop()

    embedder = Embedder(embed_model)

    def ask_fn(q: str):
        retrieved = retrieve(
            store=store,
            embed_query_fn=embedder.embed_query,
            query=q,
            top_k=top_k,
            use_mmr=use_mmr
        )
        return answer_with_optional_llm(
            question=q,
            retrieved_items=retrieved,
            use_gemini=use_gemini,
            gemini_client=gemini_client,
            gemini_model=gemini_model
        )

    report = run_eval(eval_items, ask_fn, out_dir="outputs")

    st.subheader("Summary")
    st.write({"n": report["n"], "accuracy": report["accuracy"]})

    df = pd.DataFrame(report["results"])

    st.subheader("Results")
    st.dataframe(df[["question", "expected", "answer", "score"]], use_container_width=True)

    st.subheader("Failure examples (score = 0)")
    fails = df[df["score"] < 1.0].head(20)
    if len(fails) == 0:
        st.success("No failures found for this eval set.")
    else:
        st.dataframe(fails[["question", "expected", "answer", "score"]], use_container_width=True)

    st.success("Saved report to outputs/eval_report.json")
