import streamlit as st
import uuid

from backend.config import SETTINGS
from backend.embeddings import Embedder
from backend.review_registry import ReviewRegistry
from backend.generation_usage import generation_badge
from backend.query_service import PublicDemoPolicyError, create_gemini_client, run_query
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
    if SETTINGS.low_memory_demo:
        st.error("The hosted demo corpus is temporarily unavailable. Please retry after the service finishes waking.")
    else:
        st.warning("No index found. Go to “Ingest & Index” first.")
    st.stop()

generator_label = "Gemini-assisted" if SETTINGS.low_memory_demo and SETTINGS.gemini_api_key.strip() else ("Exact extractive fallback" if SETTINGS.low_memory_demo else ("Gemini" if SETTINGS.gemini_api_key.strip() else "Extractive fallback"))
status_pills([("Index online", True), (f"{len(store.meta.get('items', [])):,} chunks", True), (generator_label, True)])

section("Try the demo", "Choose a tested prompt or ask your own question about the sanitized corpus.")
samples = [
    ("Audit retention", "How long are Aegis audit events retained?"),
    ("Incident ID", "What incident ID investigated Meridian clock skew?"),
    ("Current vs old", "Are current Aegis audit events retained for 400 or 90 days?"),
    ("Should abstain", "What is the manufacturing cost of Northstar's hardware appliance?"),
]
if "demo_question" not in st.session_state:
    st.session_state.demo_question = ""
for column, (label, sample) in zip(st.columns(4), samples):
    column.button(label, key=f"sample_{label}", use_container_width=True,
                  on_click=lambda value=sample: st.session_state.update(demo_question=value))

if SETTINGS.low_memory_demo:
    top_k = st.slider("Evidence depth", 3, SETTINGS.demo_top_k_max, min(SETTINGS.top_k, SETTINGS.demo_top_k_max), step=1)
    strategy = "lexical"
    embed_model = ""
    st.caption("Hosted profile: BM25 retrieval · no local embedding, reranker, or NLI model is loaded.")
else:
    section("Query controls", "Tune retrieval depth and diversity for this run.")
    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Top-K chunks", 3, 12, SETTINGS.top_k, step=1)
    with col2:
        strategy = st.selectbox("Retrieval strategy", ["dense_mmr", "dense", "hybrid_rrf", "hybrid_rerank", "lexical"], format_func=lambda value: {"lexical": "Lexical (BM25)", "dense": "Dense", "dense_mmr": "Dense + MMR", "hybrid_rrf": "Hybrid (Dense + BM25 + RRF)", "hybrid_rerank": "Hybrid + cross-encoder reranking"}[value])
    with col3:
        embed_model = st.text_input("Embedding model", SETTINGS.embedding_model)

question = st.text_area("Your question", key="demo_question", height=120, max_chars=SETTINGS.demo_question_max_chars if SETTINGS.low_memory_demo else 4000, placeholder="What does the evidence say about…?")

gemini_client = None
gemini_model = SETTINGS.gemini_model
review_registry = ReviewRegistry(str(tenant_dir(security) / "reviews.db"))

# Init Gemini client
if SETTINGS.gemini_api_key.strip():
    try:
        gemini_client = create_gemini_client()
    except Exception:
        st.warning("Gemini API key detected but client initialization failed.")
else:
    st.info("No Gemini API key found — using extractive fallback.")

if SETTINGS.low_memory_demo:
    st.info("Gemini may select up to two verbatim evidence sentences. If it is unavailable or over quota, the same query continues with a local exact-evidence fallback.")
    st.caption("Privacy: your question and sanitized evidence excerpts may be sent to Google's Free-tier Gemini service. Do not enter personal, confidential, or proprietary information. Raw questions are never stored.")


# If no key is actually available, requests will fail; we detect that at runtime and fallback.
if st.button("Run grounded query", type="primary", disabled=not question.strip(), use_container_width=True):
    try:
        if "demo_session_id" not in st.session_state:
            st.session_state.demo_session_id = uuid.uuid4().hex
        out = run_query(
            store=store,
            question=question,
            top_k=top_k,
            strategy=strategy,
            embedder_factory=(lambda: Embedder(embed_model)) if strategy != "lexical" else None,
            scope=RetrievalScope.from_context(security),
            review_registry=review_registry,
            client_key=st.session_state.demo_session_id,
            gemini_client=gemini_client,
            organization_id=security.organization_id,
            actor_user_id=security.user_id,
        )
    except PublicDemoPolicyError as exc:
        st.error(str(exc))
        st.stop()
    retrieved = out.pop("retrieval_result")
    generation = out.get("generation", {})
    label = generation_badge(generation)
    status_pills([(label, not generation.get("fallback_used", False))])
    if generation.get("fallback_used"):
        st.caption(f"Gemini was skipped: {generation.get('fallback_reason', 'fallback')}. Your answer was still produced locally.")

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
                    st.json(evidence_rows, expanded=False)

    section("Source citations", "Trace the response back to its highest-ranked evidence.")
    if out["citations"]:
        for citation in out["citations"]:
            with st.container(border=True):
                st.markdown(f"**{citation.get('chunk_id', 'evidence')}**")
                location = " · ".join(
                    value for value in (
                        str(citation.get("source", "")).strip(),
                        f"page {citation['page']}" if citation.get("page") is not None else "",
                    ) if value
                )
                if location:
                    st.caption(location)
                if citation.get("excerpt"):
                    st.write(citation["excerpt"])
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
    st.json(rows, expanded=False)

    with st.expander("Inspect the exact context sent to the generator"):
        st.code(out["context"][:6000])

    section("Run performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retrieval", f"{retrieved.latency_ms:.0f} ms")
    c2.metric("Generation", f"{out['latency_ms']['generation_ms']} ms")
    c3.metric("End to end", f"{out['latency_ms']['total_ms']} ms")
    c4.metric("Reranking", f"{retrieved.reranking_latency_ms:.0f} ms")
footer()
