import streamlit as st
import pandas as pd
import time
import json

from backend.config import SETTINGS
from backend.embeddings import Embedder
from backend.vectorstore import FaissStore
from backend.retriever import retrieve
from backend.reranker import RerankerUnavailableError
from backend.qa import answer_with_optional_llm
from backend.review_registry import ReviewRegistry
from backend.telemetry import log_run
from backend.utils import now_ms
from app._bootstrap import bootstrap
bootstrap()
from app.ui import configure_page, footer, page_header, section, status_pills
from app.security_ui import security_context, tenant_dir, tenant_store
from backend.security_models import RetrievalScope

configure_page("Ask & explain", "⌁")
security = security_context("query", allow_anonymous=True)
page_header("Retrieval debugger", "Ask a question. Inspect every piece of evidence.", "Trace ranked chunks, similarity scores, source pages, final context, citations, and latency for one grounded response.")

if SETTINGS.platform_mode == "postgres":
    from app.production_ui import render_ask
    render_ask()
    st.stop()

# Load index
store = tenant_store(security)
loaded = store.index is not None
if not loaded:
    st.warning("No index found. Go to “Ingest & Index” first.")
    st.stop()

status_pills([("Index online", True), (f"{len(store.meta.get('items', [])):,} chunks", True), ("Gemini" if SETTINGS.gemini_api_key.strip() else "Extractive fallback", bool(SETTINGS.gemini_api_key.strip()))])

section("Query controls", "Tune retrieval depth and diversity for this run.")
col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.slider("Top-K chunks", 3, 12, SETTINGS.top_k, step=1)
with col2:
    strategy = st.selectbox("Retrieval strategy", ["dense_mmr", "dense", "hybrid_rrf", "hybrid_rerank"], format_func=lambda value: {"dense": "Dense", "dense_mmr": "Dense + MMR", "hybrid_rrf": "Hybrid (Dense + BM25 + RRF)", "hybrid_rerank": "Hybrid + cross-encoder reranking"}[value])
with col3:
    embed_model = st.text_input("Embedding model", SETTINGS.embedding_model)

question = st.text_area("Your question", height=120, placeholder="What does the evidence say about…?")

use_gemini = True  # We default to Gemini; fallback occurs if key missing
st.caption("Generator: Gemini (falls back to extractive if no API key detected)")

gemini_client = None
gemini_model = SETTINGS.gemini_model
review_registry = ReviewRegistry(str(tenant_dir(security) / "reviews.db"))

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
if st.button("Run grounded query", type="primary", disabled=not question.strip(), use_container_width=True):
    embedder = Embedder(embed_model)

    t0 = time.time()
    try:
        retrieved = retrieve(
            store=store,
            embedder=embedder,
            query=question,
            top_k=top_k,
            strategy=strategy,
            scope=RetrievalScope.from_context(security),
        )
    except RerankerUnavailableError as exc:
        st.error(f"The local reranker could not be loaded: {exc}")
        st.stop()
    t1 = time.time()

    # Try Gemini, fallback if it errors (missing key / quota / etc.)
    try:
        out = answer_with_optional_llm(
                    question=question,
                    retrieved_items=retrieved,
                    use_gemini=use_gemini,
                    gemini_client=gemini_client,
                    gemini_model=gemini_model,
                    review_registry=review_registry,
                )
    except Exception:
        out = answer_with_optional_llm(
            question=question,
            retrieved_items=retrieved,
            use_gemini=False,
            gemini_client=None,
            gemini_model=gemini_model,
            review_registry=review_registry,
        )

    t2 = time.time()

    retrieval_ms = int((t1 - t0) * 1000)
    generation_ms = int((t2 - t1) * 1000)
    total_ms = int((t2 - t0) * 1000)

    section("Grounded answer", "Generated from the retrieved context below.")
    with st.container(border=True): st.markdown(out["answer"])

    grounding = out.get("grounding", {})
    section("Claim verification", "Every displayed claim must pass citation, deterministic, entailment, and conflict checks.")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Status", grounding.get("status", "unknown").replace("_", " ").title())
    g2.metric("Accepted", len(grounding.get("accepted_claims", [])))
    g3.metric("Removed", len(grounding.get("rejected_claims", [])))
    g4.metric("Verification", f"{grounding.get('latency_ms', {}).get('total_verification', 0):.0f} ms")
    for warning in grounding.get("warnings", []):
        st.warning(warning)
    for claim in grounding.get("accepted_claims", []):
        with st.container(border=True):
            st.markdown(f"**{claim['claim_id']} · supported** — {claim['text']}")
            st.caption(f"Entailment {claim['entailment_score']:.3f} · Contradiction {claim['contradiction_score']:.3f} · {', '.join(claim['cited_chunk_ids'])}")
    if grounding.get("rejected_claims"):
        with st.expander("Inspect claims removed by strict grounding"):
            for claim in grounding["rejected_claims"]:
                st.markdown(f"**{claim['claim_id']} · {claim['verdict']}** — {claim['text']}")
                st.caption(" · ".join(claim.get("reason_codes", [])))
                evidence_rows = [{
                    "chunk_id": item["chunk_id"], "cited": item["cited"],
                    "entailment": round(item["entailment_score"], 4),
                    "contradiction": round(item["contradiction_score"], 4),
                    "excerpt": item["excerpt"],
                } for item in claim.get("evidence", [])]
                if evidence_rows:
                    st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)

    section("Source citations", "Trace the response back to its highest-ranked evidence.")
    if out["citations"]:
        st.table(pd.DataFrame(out["citations"]))
    else:
        st.write("No citations available.")

    section("Retrieval trace", "Similarity-ranked chunks selected before generation.")
    rows = []
    for hit in retrieved.hits:
        item = hit.item
        rows.append({
            "rank": hit.rank,
            "final_score": round(hit.final_score, 5),
            "dense_rank": hit.dense_rank,
            "lexical_rank": hit.lexical_rank,
            "fusion_rank": hit.fusion_rank,
            "reranker_rank": hit.reranker_rank,
            "reranker_score": round(hit.reranker_score, 5) if hit.reranker_score is not None else None,
            "stages": " + ".join(hit.stages),
            "chunk_id": item["chunk_id"],
            "source": item["source"],
            "page": item["page"],
            "preview": item["text"][:220].replace("\n", " ") + "..."
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("Inspect the exact context sent to the generator"):
        st.code(out["context"][:6000])

    section("Run performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieval", f"{retrieved.latency_ms:.0f} ms")
    c2.metric("Generation", f"{generation_ms} ms")
    c3.metric("End to end", f"{total_ms} ms")
    c4.metric("Reranking", f"{retrieved.reranking_latency_ms:.0f} ms")

    # Log run to SQLite
    log_run({
        "ts_ms": now_ms(),
        "query": question,
        "organization_id": security.organization_id,
        "actor_user_id": security.user_id,
        "top_k": top_k,
        "use_mmr": strategy == "dense_mmr",
        "retrieval_ms": int(retrieved.latency_ms),
        "generation_ms": generation_ms,
        "total_ms": total_ms,
        "citation_count": len(out["citations"]),
        "verification_ms": int(grounding.get("latency_ms", {}).get("total_verification", 0)),
        "grounding_status": grounding.get("status", ""),
        "accepted_claims": len(grounding.get("accepted_claims", [])),
        "rejected_claims": len(grounding.get("rejected_claims", [])),
        "conflict_count": sum(1 for claim in grounding.get("rejected_claims", []) if claim.get("verdict") == "disputed"),
    })
footer()
