from pathlib import Path

import pandas as pd
import streamlit as st

from app._bootstrap import bootstrap
bootstrap()
from app.ui import configure_page, footer, page_header, section, status_pills
from app.security_ui import security_context, security_registry, tenant_dir
from backend.config import SETTINGS
from backend.document_parsers import DocumentParseError
from backend.embeddings import Embedder
from backend.ingestion import IngestionOptions, IngestionService, IngestionWorker, parser_capabilities
from backend.ingestion_registry import IngestionRegistry
from backend.query_service import create_generation_client

configure_page("Ingest & index", "⬡")
security = security_context("documents:write")
if security.organization_id == SETTINGS.public_organization_id:
    st.error("The public demo corpus is read-only.")
    st.stop()
page_header(
    "Knowledge pipeline",
    "Build versioned, context-aware evidence.",
    "Parse structured documents, preserve headings and tables, track every ingestion stage, and update the active index without stale chunks.",
)

if SETTINGS.platform_mode == "postgres":
    from app.production_ui import render_ingestion
    render_ingestion()
    st.stop()


@st.cache_resource
def ingestion_runtime(organization_id, actor_user_id):
    root = tenant_dir(security)
    registry = IngestionRegistry(str(root / "lifecycle.db"), organization_id)
    client = None
    if SETTINGS.groq_api_key.strip():
        try:
            client = create_generation_client()
        except Exception:
            pass
    service = IngestionService(registry, str(root), SETTINGS.uploads_dir, Embedder, client,
                               organization_id, actor_user_id, security_registry())
    worker = IngestionWorker(service, SETTINGS.ingestion_worker_lease_seconds)
    worker.start()
    return registry, service, worker, client is not None


registry, service, worker, provider_available = ingestion_runtime(security.organization_id, security.user_id)
capabilities = parser_capabilities()
status_pills([
    ("PDF layout", capabilities["pdf"]),
    ("Tables", capabilities["tables"]),
    ("DOCX + HTML", capabilities["docx"] and capabilities["html"]),
    ("OCR", capabilities["ocr"]),
    ("Groq context", provider_available),
])
if not capabilities["ocr"]:
    st.caption("OCR is optional. Text PDFs and other formats remain available; scanned-only PDFs finish with an actionable warning or OCR_REQUIRED error.")

section("Add or update documents", "Uploading the same logical filename updates its version; identical content is detected as unchanged.")
use_demo = st.toggle("Use bundled 60-card sanitized corpus", value=False)
uploaded = st.file_uploader(
    "Upload up to 10 documents",
    type=["pdf", "docx", "md", "markdown", "html", "htm", "txt"],
    accept_multiple_files=True,
    disabled=use_demo,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    child_tokens = st.slider("Child tokens", 100, 1000, SETTINGS.chunk_tokens, 20)
with c2:
    overlap_tokens = st.slider("Overlap", 0, min(300, child_tokens - 1), min(SETTINGS.chunk_overlap, child_tokens - 1), 10)
with c3:
    parent_tokens = st.slider("Parent tokens", child_tokens, 2400, max(SETTINGS.parent_tokens, child_tokens), 100)
with c4:
    context_label = st.selectbox("Context mode", ["Deterministic", "Groq enhanced"], disabled=not provider_available)

source_version = st.text_input("Source version label (optional)", placeholder="For example: 2026.08 or current")
context_mode = "provider" if context_label == "Groq enhanced" and provider_available else "deterministic"
options = IngestionOptions(
    child_tokens=child_tokens,
    overlap_tokens=overlap_tokens,
    parent_tokens=parent_tokens,
    embedding_model=SETTINGS.embedding_model,
    context_mode=context_mode,
    context_model=SETTINGS.context_model,
    context_prompt_version=SETTINGS.context_prompt_version,
    source_version=source_version.strip() or None,
)

demo_paths = sorted(Path("data/public_demo").glob("*.md")) if use_demo else []
selected_count = len(demo_paths) if use_demo else len(uploaded or [])
if selected_count:
    st.caption(f"{selected_count} document{'s' if selected_count != 1 else ''} selected")

if st.button("Queue context-aware ingestion", type="primary", disabled=selected_count == 0, use_container_width=True):
    if not use_demo and selected_count > SETTINGS.max_documents_per_job:
        st.error(f"Upload at most {SETTINGS.max_documents_per_job} documents at once.")
        st.stop()
    queued, failures = [], []
    sources = [(path.name, path.read_bytes()) for path in demo_paths] if use_demo else [(item.name, item.getvalue()) for item in uploaded]
    for filename, content in sources:
        try:
            queued.append(service.stage_upload(filename, content, options))
        except DocumentParseError as exc:
            failures.append(f"{filename}: {exc.code} — {exc}")
    worker.start()
    if queued:
        st.success(f"Queued {len(queued)} ingestion job{'s' if len(queued) != 1 else ''}. Use Refresh status below to follow progress.")
    for failure in failures:
        st.error(failure)

section("Job monitor", "Durable SQLite jobs retain stage progress, warnings, failures, and retry attempts across app reruns.")
if st.button("Refresh status", use_container_width=True):
    st.rerun()

jobs = registry.list_jobs(50)
if jobs:
    st.dataframe(pd.DataFrame([{
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": f"{job['progress'] * 100:.0f}%",
        "attempts": job["attempts"],
        "outcome": (job["result"] or {}).get("outcome", ""),
        "error": job["error_code"] or "",
    } for job in jobs]), use_container_width=True, hide_index=True)
    detail_id = st.selectbox("Inspect job details", [job["job_id"] for job in jobs])
    detail = registry.get_job(detail_id)
    with st.expander("Job events, warnings, and result"):
        st.json({"status": detail["status"], "warnings": detail["warnings"], "error": {"code": detail["error_code"], "message": detail["error_message"]}, "result": detail["result"], "events": detail["events"]})
    queued_jobs = [job for job in jobs if job["status"] == "queued"]
    if queued_jobs:
        cancel_id = st.selectbox("Queued job to cancel", [job["job_id"] for job in queued_jobs])
        if st.button("Cancel selected job") and registry.cancel_job(cancel_id):
            st.rerun()
    failed_jobs = [job for job in jobs if job["status"] == "failed"]
    if failed_jobs:
        retry_id = st.selectbox("Failed job to retry", [job["job_id"] for job in failed_jobs])
        if st.button("Retry selected job"):
            if registry.retry_job(retry_id, SETTINGS.ingestion_retry_limit):
                worker.start()
                st.success("Job queued for retry.")
                st.rerun()
            else:
                st.error("Retry limit reached or the job is no longer failed.")
else:
    st.info("No ingestion jobs have been created yet.")

section("Document lifecycle", "Active versions contribute chunks to FAISS; deleted documents are immediately removed from retrieval.")
documents = registry.list_documents()
if documents:
    st.dataframe(pd.DataFrame([{
        "document_id": document["document_id"],
        "source": document["logical_source"],
        "version": document.get("source_version"),
        "active": bool(document["active"]),
        "chunks": document["chunk_count"],
        "checksum": (document.get("source_sha256") or "")[:12],
    } for document in documents]), use_container_width=True, hide_index=True)
    active_documents = [document for document in documents if document["active"]]
    if active_documents:
        selected_document = st.selectbox(
            "Inspect active document",
            [document["document_id"] for document in active_documents],
            format_func=lambda value: next(document["logical_source"] for document in active_documents if document["document_id"] == value),
        )
        chunks = registry.document_chunks(selected_document)
        if chunks:
            st.dataframe(pd.DataFrame([{
                "chunk_id": chunk["chunk_id"],
                "heading": " > ".join(chunk.get("heading_path", [])),
                "types": ", ".join(chunk.get("block_types", [])),
                "pages": f"{chunk.get('page_start')}-{chunk.get('page_end')}",
                "context": chunk.get("contextual_prefix", ""),
                "preview": chunk["text"][:180],
            } for chunk in chunks]), use_container_width=True, hide_index=True)
        if st.button("Soft-delete selected document", type="secondary"):
            if service.delete_document(selected_document):
                st.success("Document and all of its chunks were removed from the active index. Version history remains in the registry.")
                st.rerun()
            st.error("Document was not active.")
else:
    st.info("No lifecycle-managed documents yet. Legacy indexes remain queryable until you build this profile.")

manifest_path = Path(SETTINGS.index_dir) / "manifest.json"
if manifest_path.exists():
    with st.expander("Active index manifest"):
        st.json(__import__("json").loads(manifest_path.read_text(encoding="utf-8")))

footer()
