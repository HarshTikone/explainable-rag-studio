"""API-only Streamlit views used when PLATFORM_MODE=postgres."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from app.api_client import ApiClientError, RagApiClient
from app.ui import footer, section, status_pills
from backend.config import SETTINGS


def client() -> RagApiClient:
    session_cookie = ""
    try:
        session_cookie = st.context.cookies.get("rag_session", "")
    except Exception:
        pass
    return RagApiClient(st.session_state.get("security_api_key", ""), session_cookie=session_cookie)


def render_ask() -> None:
    status_pills([("API isolated", True), ("PostgreSQL + pgvector", True), ("Strict grounding", True)])
    section("Query controls", "The frontend sends the query to the authenticated tenant API.")
    c1, c2 = st.columns(2)
    with c1:
        top_k = st.slider("Top-K chunks", 3, 12, SETTINGS.top_k)
    with c2:
        strategy = st.selectbox("Retrieval strategy", ["dense_mmr", "dense", "hybrid_rrf", "hybrid_rerank"])
    question = st.text_area("Your question", height=120)
    if st.button("Run grounded query", type="primary", disabled=not question.strip(), use_container_width=True):
        try:
            result = client().ask(question, top_k, strategy)
        except ApiClientError as exc:
            st.error(str(exc)); footer(); return
        section("Grounded answer", "Only claims accepted by strict verification are displayed.")
        with st.container(border=True):
            st.markdown(result["answer"])
        grounding = result.get("grounding", {})
        cols = st.columns(4)
        cols[0].metric("Status", grounding.get("status", "unknown").replace("_", " ").title())
        cols[1].metric("Accepted", len(grounding.get("accepted_claims", [])))
        cols[2].metric("Removed", len(grounding.get("rejected_claims", [])))
        cols[3].metric("Total", f"{result.get('latency_ms', {}).get('total_ms', 0)} ms")
        section("Citations", "Citations are derived only from accepted claims.")
        st.dataframe(pd.DataFrame(result.get("citations", [])), use_container_width=True, hide_index=True)
        section("Retrieval trace", "Stage-specific ranks and scores returned by the API.")
        st.dataframe(pd.DataFrame(result.get("retrieved", [])), use_container_width=True, hide_index=True)
        if grounding.get("rejected_claims"):
            with st.expander("Claims removed by strict grounding"):
                st.json(grounding["rejected_claims"])
    footer()


def render_ingestion() -> None:
    status_pills([("Distributed workers", True), ("Encrypted object storage", True), ("Tenant RLS", True)])
    uploaded = st.file_uploader("Upload documents", type=["pdf", "docx", "md", "markdown", "html", "htm", "txt"], accept_multiple_files=True)
    c1, c2, c3 = st.columns(3)
    child = c1.slider("Child tokens", 100, 1000, SETTINGS.chunk_tokens, 20)
    overlap = c2.slider("Overlap", 0, min(300, child - 1), min(SETTINGS.chunk_overlap, child - 1), 10)
    parent = c3.slider("Parent tokens", child, 2400, max(SETTINGS.parent_tokens, child), 100)
    if st.button("Queue secure ingestion", type="primary", disabled=not uploaded, use_container_width=True):
        files = [(item.name, item.getvalue(), item.type or "application/octet-stream") for item in uploaded]
        try:
            result = client().upload(files, {"context_mode": "deterministic", "child_tokens": child,
                                             "overlap_tokens": overlap, "parent_tokens": parent})
            st.success("Upload validation completed.")
            st.json(result)
        except ApiClientError as exc:
            st.error(str(exc))
    section("Document inventory", "All records are filtered by PostgreSQL row-level security.")
    try:
        documents = client().documents().get("documents", [])
        st.dataframe(pd.DataFrame(documents), use_container_width=True, hide_index=True)
        active = [row for row in documents if row.get("active")]
        if active:
            selected = st.selectbox("Document to soft-delete", [row["document_id"] for row in active])
            if st.button("Soft-delete selected document"):
                client().delete_document(selected); st.rerun()
    except ApiClientError as exc:
        st.error(str(exc))
    footer()


def render_reviews() -> None:
    status = st.selectbox("Case status", ["open", "resolved", "all"])
    try:
        cases = client().reviews(status).get("cases", [])
    except ApiClientError as exc:
        st.error(str(exc)); footer(); return
    status_pills([("PostgreSQL queue", True), (f"{len(cases)} cases", bool(cases)), ("Tenant scoped", True)])
    if cases:
        st.dataframe(pd.DataFrame(cases), use_container_width=True, hide_index=True)
        selected_id = st.selectbox("Review case", [row["case_id"] for row in cases])
        selected = client().review(selected_id)
        st.json(selected)
        if selected.get("status") == "open":
            decision = st.radio("Human verdict", ["supported", "unsupported", "contradicted"], horizontal=True)
            notes = st.text_area("Notes")
            if st.button("Save decision", type="primary"):
                client().decide(selected_id, decision, notes); st.rerun()
    else:
        st.info("No review cases match this filter.")
    footer()


def render_security() -> None:
    api = client()
    try:
        chain = api.verify_audit()
        quarantine = api.quarantine().get("cases", [])
        keys = api.keys().get("api_keys", [])
    except ApiClientError as exc:
        st.error(str(exc)); footer(); return
    status_pills([("PostgreSQL RLS", True), ("Audit chain valid", chain.get("valid", False)),
                  (f"{len(quarantine)} quarantine cases", not quarantine)])
    section("API keys", "Secrets are never returned after creation.")
    st.dataframe(pd.DataFrame(keys), use_container_width=True, hide_index=True)
    section("Quarantine", "Quarantined objects never enter retrieval.")
    st.dataframe(pd.DataFrame(quarantine), use_container_width=True, hide_index=True)
    section("Audit verification", "The API verifies this organization's HMAC chain.")
    st.json(chain)
    try:
        audit = api.audit()
        st.download_button("Export redacted audit JSONL", data="\n".join(json.dumps(row, default=str, sort_keys=True) for row in audit.get("events", [])),
                           file_name="audit-redacted.jsonl", mime="application/x-ndjson")
    except ApiClientError:
        pass
    footer()
