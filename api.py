"""Authenticated, tenant-isolated FastAPI surface."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import List, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from google import genai

from backend.config import SETTINGS
from backend.document_parsers import DocumentParseError
from backend.embeddings import Embedder
from backend.grounding import get_default_verifier
from backend.grounding_policy import default_grounding_policy
from backend.ingestion import IngestionOptions, IngestionService, IngestionWorker, QuarantinedUpload, parser_capabilities
from backend.ingestion_registry import IngestionRegistry
from backend.qa import answer_with_optional_llm
from backend.reranker import RerankerUnavailableError
from backend.retriever import retrieve
from backend.review_registry import ReviewRegistry
from backend.schemas import ApiKeyCreateRequest, AskRequest, MembershipCreateRequest, MembershipUpdateRequest, OidcIdentityRequest, QuarantineDecisionRequest, ReviewDecisionRequest
from backend.security import SecurityRegistry
from backend.security_models import AuthenticationError, AuthorizationError, ROLE_SCOPES, RetrievalScope, SecurityBoundaryError, SecurityContext
from backend.security_scanner import scanner_capabilities
from backend.tenant_store import TenantStoreManager
from backend.vectorstore import FaissStore
from backend.platform_runtime import PlatformConfigurationError, get_platform_runtime
from backend.postgres_security import PostgresSecurityRegistry
from backend.rate_limit import MemoryDemoRateLimiter, RateLimitUnavailable
from backend.postgres_review import PostgresReviewRegistry


platform_runtime = get_platform_runtime()
security_registry = (
    PostgresSecurityRegistry(platform_runtime.database, SETTINGS.api_key_pepper, SETTINGS.audit_hmac_key)
    if platform_runtime else SecurityRegistry(SETTINGS.security_db_path, SETTINGS.api_key_pepper, SETTINGS.audit_hmac_key)
)
security_registry.ensure_public_organization(SETTINGS.public_organization_id)
tenant_stores = TenantStoreManager(SETTINGS.tenant_index_root, SETTINGS.tenant_store_cache_size)
store = FaissStore(SETTINGS.index_dir)  # explicit read-only bridge for the legacy public demo
store.load()
ingestion_registry = IngestionRegistry(SETTINGS.ingestion_db_path, SETTINGS.public_organization_id)  # compatibility
review_registry = ReviewRegistry(SETTINGS.review_db_path)  # compatibility
gemini_client = genai.Client(api_key=SETTINGS.gemini_api_key) if SETTINGS.gemini_api_key.strip() else None
ingestion_service = IngestionService(ingestion_registry, SETTINGS.index_dir, SETTINGS.uploads_dir, Embedder, gemini_client)
ingestion_worker = IngestionWorker(ingestion_service, SETTINGS.ingestion_worker_lease_seconds)
embedder = None
_runtimes = {}
_runtime_lock = RLock()
_demo_rate_limiter = MemoryDemoRateLimiter()


def get_embedder():
    global embedder
    if embedder is None:
        embedder = Embedder(SETTINGS.embedding_model)
    return embedder


def _anonymous() -> SecurityContext:
    return SecurityContext(SETTINGS.public_organization_id, "anonymous_demo", "viewer", None, ROLE_SCOPES["viewer"], True)


def _runtime(context: SecurityContext):
    if platform_runtime:
        raise RuntimeError("Local tenant runtimes are unavailable in PostgreSQL mode.")
    with _runtime_lock:
        value = _runtimes.get(context.organization_id)
        if value:
            value[1].actor_user_id = context.user_id
            return value
        root = tenant_stores.organization_dir(context.organization_id)
        registry = IngestionRegistry(str(root / "lifecycle.db"), context.organization_id)
        reviews = ReviewRegistry(str(root / "reviews.db"))
        service = IngestionService(registry, str(root), SETTINGS.uploads_dir, Embedder, gemini_client,
                                   context.organization_id, context.user_id, security_registry)
        value = (registry, service, IngestionWorker(service, SETTINGS.ingestion_worker_lease_seconds), reviews)
        _runtimes[context.organization_id] = value
        return value


def _store(context: SecurityContext):
    if platform_runtime:
        return platform_runtime.vector_store(context)
    tenant = tenant_stores.get(context.organization_id, reload=True)
    if tenant.index is None and context.organization_id == SETTINGS.public_organization_id:
        store.load()
        return store
    return tenant


def _session_context(request: Request | None) -> SecurityContext | None:
    if not platform_runtime or request is None:
        return None
    session_id = request.cookies.get("rag_session")
    if not session_id:
        return None
    try:
        import json as _json
        raw = platform_runtime.redis.get(f"rag:session:{session_id}")
        if not raw:
            return None
        value = _json.loads(raw)
        return SecurityContext(value["organization_id"], value["user_id"], value["role"], None, frozenset(value["scopes"]))
    except Exception:
        return None


def _rate_limit(context: SecurityContext, request: Request | None) -> None:
    path = request.url.path if request is not None else "/direct"
    if path in {"/health", "/ready"}:
        return
    limiter = platform_runtime.rate_limiter if platform_runtime else (_demo_rate_limiter if context.anonymous_demo else None)
    if limiter is None:
        return
    limit, window = SETTINGS.rate_limit_query_per_minute, 60
    if path.startswith("/ingestion/jobs") and (request is None or request.method == "POST"):
        limit, window = SETTINGS.rate_limit_ingest_per_hour, 3600
    elif request is not None and request.method not in {"GET", "HEAD", "OPTIONS"}:
        limit = SETTINGS.rate_limit_write_per_minute
    identity = context.key_id or context.user_id
    key = f"{context.organization_id}:{identity}:{request.method if request else 'DIRECT'}:{path.split('/')[1]}"
    try:
        decision = limiter.check(key, limit, window)
    except RateLimitUnavailable as exc:
        raise HTTPException(503, "Rate-limit enforcement is unavailable.") from exc
    if not decision.allowed:
        raise HTTPException(429, "Rate limit exceeded.", headers={"Retry-After": str(decision.reset_after_seconds), "X-RateLimit-Limit": str(decision.limit), "X-RateLimit-Remaining": "0"})
    if request is not None:
        request.state.rate_limit = decision


def context_from_bearer(request: Request = None, authorization: str | None = Header(default=None)) -> SecurityContext:
    if not authorization:
        session = _session_context(request)
        if session:
            _rate_limit(session, request)
            return session
        if SETTINGS.security_mode == "demo":
            context = _anonymous()
            _rate_limit(context, request)
            return context
        if not security_registry.protected_ready:
            raise HTTPException(503, "Protected security configuration is incomplete.")
        raise HTTPException(401, "Bearer credentials are required.", headers={"WWW-Authenticate": "Bearer"})
    scheme, _, credential = authorization.partition(" ")
    if scheme.casefold() != "bearer" or not credential:
        raise HTTPException(401, "Invalid credentials.")
    try:
        if platform_runtime and credential.count(".") == 2:
            principal = platform_runtime.oidc.validate(credential)
            context = security_registry.authenticate_oidc(principal)
        else:
            context = security_registry.authenticate(credential)
        _rate_limit(context, request)
        return context
    except AuthenticationError as exc:
        if security_registry.protected_ready:
            try:
                security_registry.append_audit("unknown", None, None, "authentication.failed", "api_key", None, "denied", 401, "INVALID_CREDENTIALS")
            except Exception:
                pass
        raise HTTPException(401, "Invalid credentials.") from exc


def _require(context: SecurityContext, scope: str, anonymous: bool = False) -> None:
    if context.anonymous_demo and not anonymous:
        raise HTTPException(403, "An authenticated API key is required.")
    try:
        context.require(scope)
    except AuthorizationError as exc:
        raise HTTPException(403, "Forbidden.") from exc


def _audit(context: SecurityContext, action: str, kind: str, object_id=None, status=200, result="success", reason="", details=None, request_id="") -> str:
    if context.anonymous_demo and not security_registry.protected_ready:
        return ""
    try:
        return security_registry.append_audit(context.organization_id, context.user_id, context.key_id, action, kind,
                                              object_id, result, status, reason, request_id, details)
    except Exception as exc:
        raise HTTPException(503, "Security audit storage is unavailable.") from exc


@asynccontextmanager
async def lifespan(_app):
    yield
    for _, _, worker, _ in _runtimes.values():
        worker.stop()


app = FastAPI(title="Explainable RAG API", lifespan=lifespan)


@app.exception_handler(AuthorizationError)
async def authorization_error(_request: Request, _exc: AuthorizationError):
    return JSONResponse(status_code=403, content={"detail": "Forbidden."})


@app.middleware("http")
async def request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID", "req_" + uuid.uuid4().hex)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    decision = getattr(request.state, "rate_limit", None)
    if decision:
        response.headers["X-RateLimit-Limit"] = str(decision.limit)
        response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        response.headers["X-RateLimit-Reset"] = str(decision.reset_after_seconds)
    return response


@app.get("/auth/me")
def auth_me(context: SecurityContext = Depends(context_from_bearer)):
    if context.anonymous_demo:
        raise HTTPException(401, "Bearer credentials are required.")
    return {"organization_id": context.organization_id, "user_id": context.user_id, "role": context.role,
            "key_id": context.key_id, "scopes": sorted(context.scopes),
            "authentication_method": "api_key" if context.key_id else "oidc",
            "quotas": {"documents": SETTINGS.quota_documents, "upload_bytes": SETTINGS.quota_upload_bytes,
                       "concurrent_jobs": SETTINGS.quota_concurrent_jobs}}


@app.get("/auth/login")
def oidc_login():
    if not platform_runtime:
        raise HTTPException(404, "OIDC is available only in production-platform mode.")
    try:
        value = platform_runtime.pkce.begin(platform_runtime.oidc, SETTINGS.oidc_client_id, SETTINGS.oidc_redirect_uri)
    except Exception as exc:
        raise HTTPException(503, "OIDC login is unavailable.") from exc
    return RedirectResponse(value["authorization_url"], status_code=302)


@app.get("/auth/callback")
def oidc_callback(code: str, state: str):
    if not platform_runtime:
        raise HTTPException(404, "OIDC is available only in production-platform mode.")
    try:
        pending = platform_runtime.pkce.consume(state)
        client_secret = Path(SETTINGS.oidc_client_secret_file).read_text(encoding="utf-8").strip() if SETTINGS.oidc_client_secret_file else ""
        token_payload = {
            "grant_type": "authorization_code", "code": code, "redirect_uri": pending["redirect_uri"],
            "client_id": SETTINGS.oidc_client_id,
            "code_verifier": pending["verifier"],
        }
        if client_secret:
            token_payload["client_secret"] = client_secret
        token_response = platform_runtime.oidc.http.post(platform_runtime.oidc.token_endpoint, data=token_payload)
        token_response.raise_for_status()
        tokens = token_response.json()
        context = security_registry.authenticate_oidc(platform_runtime.oidc.validate(tokens["id_token"], pending["nonce"]))
        session_id = uuid.uuid4().hex
        platform_runtime.redis.setex(f"rag:session:{session_id}", 3600, json.dumps({
            "organization_id": context.organization_id, "user_id": context.user_id,
            "role": context.role, "scopes": sorted(context.scopes),
        }))
    except Exception as exc:
        raise HTTPException(401, "OIDC authentication failed.") from exc
    response = RedirectResponse(SETTINGS.app_public_url, status_code=302)
    response.set_cookie("rag_session", session_id, max_age=3600, httponly=True,
                        secure=SETTINGS.app_public_url.startswith("https://"), samesite="lax")
    return response


@app.post("/auth/refresh")
def refresh_session(context: SecurityContext = Depends(context_from_bearer)):
    return {"status": "active", "user_id": context.user_id, "expires_in": 3600}


@app.post("/auth/logout", status_code=204)
def logout(request: Request, response: Response):
    if platform_runtime:
        session_id = request.cookies.get("rag_session")
        if session_id:
            platform_runtime.redis.delete(f"rag:session:{session_id}")
    response.delete_cookie("rag_session")


@app.post("/ask")
def ask(req: AskRequest, request: Request = None, context: SecurityContext = Depends(context_from_bearer)):
    direct = not isinstance(context, SecurityContext)
    if direct:
        context = _anonymous()
    _require(context, "query", anonymous=True)
    active_store = store if direct else _store(context)
    if active_store.index is None:
        raise HTTPException(503, "Vector index is not ready for this organization.")
    started = time.perf_counter()
    try:
        found = retrieve(active_store, get_embedder(), req.question, req.top_k, req.retrieval_strategy,
                         rerank_candidates=req.rerank_candidates,
                         scope=RetrievalScope.from_context(context, getattr(getattr(request, "state", None), "request_id", "direct")))
    except SecurityBoundaryError as exc:
        _audit(context, "retrieval.boundary_violation", "index", context.organization_id, 503, "blocked", "CROSS_TENANT_METADATA")
        raise HTTPException(503, "The organization index failed its isolation check.") from exc
    except RerankerUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    retrieved_at = time.perf_counter()
    if any(hit.item.get("organization_id", SETTINGS.public_organization_id) != context.organization_id for hit in found.hits):
        raise HTTPException(503, "The organization index failed its isolation check.")
    reviews = (
        review_registry if direct else
        PostgresReviewRegistry(platform_runtime.database, context) if platform_runtime else
        _runtime(context)[3]
    )
    try:
        output = answer_with_optional_llm(req.question, found, bool(gemini_client), gemini_client, SETTINGS.gemini_model, review_registry=reviews)
    except Exception:
        output = answer_with_optional_llm(req.question, found, False, None, SETTINGS.gemini_model, review_registry=reviews)
    allowed = {hit.item.get("chunk_id") for hit in found.hits}
    if any(item.get("chunk_id") not in allowed for item in output.get("citations", [])):
        raise HTTPException(503, "Citation boundary validation failed.")
    finished = time.perf_counter()
    event = _audit(context, "query.complete", "retrieval", request_id=getattr(getattr(request, "state", None), "request_id", "direct"), details={
        "query_sha256": hashlib.sha256(req.question.encode()).hexdigest(), "query_length": len(req.question),
        "strategy": req.retrieval_strategy, "candidate_count": found.candidate_count, "returned_count": len(found.hits),
    })
    if platform_runtime and not direct:
        from backend.database import TelemetryRecord
        with platform_runtime.database.session(context, write=True) as session:
            session.add(TelemetryRecord(
                organization_id=context.organization_id, telemetry_id="tel_" + uuid.uuid4().hex,
                request_id=getattr(getattr(request, "state", None), "request_id", "direct"),
                query_sha256=hashlib.sha256(req.question.encode()).hexdigest(), query_length=len(req.question),
                category="query", metrics_json={"strategy": req.retrieval_strategy,
                    "retrieval_ms": int((retrieved_at-started)*1000), "total_ms": int((finished-started)*1000),
                    "citation_count": len(output.get("citations", [])),
                    "grounding_status": output.get("grounding", {}).get("status", "")},
            ))
    return {"answer": output["answer"], "citations": output["citations"], "retrieved": [hit.to_dict() for hit in found.hits],
            "grounding": output.get("grounding", {}),
            "latency_ms": {"retrieval_ms": int((retrieved_at-started)*1000), "generation_ms": int((finished-retrieved_at)*1000),
                           "verification_ms": int(output.get("grounding", {}).get("latency_ms", {}).get("total_verification", 0)),
                           "total_ms": int((finished-started)*1000)},
            "security": {"mode": SETTINGS.security_mode, "role": context.role, "organization_isolated": True, "audit_event_id": event}}


@app.post("/ingestion/jobs", status_code=202)
async def create_ingestion_jobs(files: List[UploadFile] = File(...), context_mode: Literal["deterministic", "gemini"] = Form("deterministic"),
                                source_version: str | None = Form(None), child_tokens: int = Form(SETTINGS.chunk_tokens),
                                overlap_tokens: int = Form(SETTINGS.chunk_overlap), parent_tokens: int = Form(SETTINGS.parent_tokens),
                                context: SecurityContext = Depends(context_from_bearer)):
    direct = not isinstance(context, SecurityContext)
    if direct:
        context = _anonymous()
    if not direct:
        _require(context, "documents:write")
    if context.organization_id == SETTINGS.public_organization_id and not direct:
        raise HTTPException(403, "The public organization cannot accept uploads.")
    if not 1 <= len(files) <= SETTINGS.max_documents_per_job:
        raise HTTPException(422, "Invalid number of documents.")
    if not 100 <= child_tokens <= 2000 or not 0 <= overlap_tokens < child_tokens or parent_tokens < child_tokens:
        raise HTTPException(422, "Invalid chunk settings.")
    if direct:
        service, worker = ingestion_service, ingestion_worker
    else:
        _, service, worker, _ = _runtime(context)
    options = IngestionOptions(child_tokens, overlap_tokens, parent_tokens, SETTINGS.embedding_model, context_mode,
                               SETTINGS.context_model, SETTINGS.context_prompt_version, source_version)
    if platform_runtime and not direct:
        jobs, quarantined = [], []
        for upload in files:
            try:
                result = platform_runtime.ingestion.enqueue_upload(
                    context, upload.filename or "document", await upload.read(),
                    upload.content_type or "application/octet-stream", options,
                )
            except DocumentParseError as exc:
                status = 413 if exc.code == "FILE_TOO_LARGE" else 422
                raise HTTPException(status, {"code": exc.code, "message": str(exc)}) from exc
            if result["status"] == "quarantined":
                quarantined.append(result)
            else:
                jobs.append(result)
        _audit(context, "ingestion.queue", "ingestion_job", jobs[0]["job_id"] if len(jobs) == 1 else None,
               status=202, details={"job_count": len(jobs), "quarantine_count": len(quarantined)})
        return {"jobs": jobs, "quarantined": quarantined}
    jobs, quarantined = [], []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.casefold()
        if suffix not in {".pdf", ".docx", ".html", ".htm", ".md", ".markdown", ".txt"}:
            raise HTTPException(415, "Unsupported document type.")
        try:
            if direct:
                job = service.stage_upload(upload.filename or "document", await upload.read(), options)
            else:
                job = service.stage_upload(upload.filename or "document", await upload.read(), options, upload.content_type)
            jobs.append({"job_id": job, "filename": Path(upload.filename or "document").name})
        except QuarantinedUpload as exc:
            quarantined.append({"case_id": exc.case_id, "filename": Path(upload.filename or "document").name})
        except DocumentParseError as exc:
            raise HTTPException(413 if exc.code == "FILE_TOO_LARGE" else 415, {"code": exc.code, "message": str(exc)}) from exc
    if jobs:
        worker.start()
    _audit(context, "ingestion.queue", "ingestion_job", jobs[0]["job_id"] if len(jobs) == 1 else None,
           status=202, details={"job_count": len(jobs), "quarantine_count": len(quarantined)})
    return {"jobs": jobs, "quarantined": quarantined}


@app.get("/ingestion/jobs/{job_id}")
def get_job(job_id: str, context: SecurityContext = Depends(context_from_bearer)):
    _require(context, "documents:read")
    if platform_runtime:
        from sqlalchemy import select
        from backend.database import IngestionJob, JobEvent
        with platform_runtime.database.session(context) as session:
            job = session.get(IngestionJob, (context.organization_id, job_id))
            if not job:
                raise HTTPException(404, "Ingestion job not found.")
            events = session.execute(select(JobEvent).where(JobEvent.job_id == job_id).order_by(JobEvent.created_at)).scalars().all()
            return {"job_id": job.job_id, "status": job.state, "attempts": job.attempts,
                    "progress": job.progress_json, "error_code": job.error_code,
                    "events": [{"stage": event.stage, "progress": event.progress, "message": event.message,
                                "created_at": event.created_at.isoformat()} for event in events]}
    job = _runtime(context)[0].get_job(job_id)
    if not job:
        raise HTTPException(404, "Ingestion job not found.")
    return job


def get_ingestion_job(job_id: str, context=None):
    """Backward-compatible direct-call adapter; HTTP requests use get_job."""
    if not isinstance(context, SecurityContext):
        job = ingestion_registry.get_job(job_id)
        if not job:
            raise HTTPException(404, "Ingestion job not found.")
        return job
    return get_job(job_id, context)


@app.post("/ingestion/jobs/{job_id}/retry", status_code=202)
def retry_job(job_id: str, context: SecurityContext = Depends(context_from_bearer)):
    _require(context, "documents:write")
    if platform_runtime:
        from backend.database import IngestionJob
        with platform_runtime.database.session(context, write=True) as session:
            job = session.get(IngestionJob, (context.organization_id, job_id))
            if not job or (context.role == "editor" and job.created_by != context.user_id):
                raise HTTPException(404, "Ingestion job not found.")
            if job.state != "failed" or job.attempts >= SETTINGS.ingestion_retry_limit:
                raise HTTPException(409, "Job is not retryable.")
            job.state, job.error_code = "queued", None
        platform_runtime.queue.enqueue(job_id)
        return {"job_id": job_id, "status": "queued"}
    registry, _, worker, _ = _runtime(context)
    job = registry.get_job(job_id)
    if not job or (context.role == "editor" and job.get("created_by") != context.user_id):
        raise HTTPException(404, "Ingestion job not found.")
    if not registry.retry_job(job_id, SETTINGS.ingestion_retry_limit):
        raise HTTPException(409, "Job is not retryable.")
    worker.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/ingestion/jobs/{job_id}/cancel")
def cancel_job(job_id: str, context: SecurityContext = Depends(context_from_bearer)):
    _require(context, "documents:write")
    if platform_runtime:
        from backend.database import IngestionJob
        with platform_runtime.database.session(context, write=True) as session:
            job = session.get(IngestionJob, (context.organization_id, job_id))
            if not job or (context.role == "editor" and job.created_by != context.user_id):
                raise HTTPException(404, "Ingestion job not found.")
            if job.state not in {"queued", "running"}:
                raise HTTPException(409, "Only queued or running jobs can be cancelled.")
            job.state = "cancelled"
        platform_runtime.queue.cancel(job_id)
        return {"job_id": job_id, "status": "cancelled"}
    registry = _runtime(context)[0]
    job = registry.get_job(job_id)
    if not job or (context.role == "editor" and job.get("created_by") != context.user_id):
        raise HTTPException(404, "Ingestion job not found.")
    if not registry.cancel_job(job_id):
        raise HTTPException(409, "Only queued jobs can be cancelled.")
    return {"job_id": job_id, "status": "cancelled"}


def cancel_ingestion_job(job_id: str, context=None):
    if not isinstance(context, SecurityContext):
        if not ingestion_registry.cancel_job(job_id):
            raise HTTPException(409, "Only queued jobs can be cancelled.")
        return {"job_id": job_id, "status": "cancelled"}
    return cancel_job(job_id, context)


@app.get("/documents")
def documents(context: SecurityContext = Depends(context_from_bearer)):
    _require(context, "documents:read")
    if platform_runtime:
        from sqlalchemy import select
        from backend.database import Document, DocumentVersion
        with platform_runtime.database.session(context) as session:
            rows = session.execute(
                select(Document, DocumentVersion)
                .outerjoin(DocumentVersion, (DocumentVersion.document_id == Document.document_id) & (DocumentVersion.active.is_(True)))
                .order_by(Document.created_at.desc())
            ).all()
            return {"documents": [{"document_id": document.document_id, "title": document.title,
                                    "logical_source": document.logical_source, "trust_state": document.trust_state,
                                    "active": document.active, "uploaded_by": document.uploaded_by,
                                    "current_version_id": version.document_version_id if version else None,
                                    "source_version": version.source_version if version else None}
                                   for document, version in rows]}
    return {"documents": _runtime(context)[0].list_documents()}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, context: SecurityContext = Depends(context_from_bearer)):
    if not isinstance(context, SecurityContext):
        if not ingestion_service.delete_document(document_id):
            raise HTTPException(404, "Active document not found.")
        return {"document_id": document_id, "status": "deleted"}
    _require(context, "documents:write")
    if platform_runtime:
        from datetime import datetime, timezone
        from sqlalchemy import update
        from backend.database import Chunk, Document, DocumentVersion
        with platform_runtime.database.session(context, write=True) as session:
            document = session.get(Document, (context.organization_id, document_id))
            if not document or not document.active or (context.role == "editor" and document.uploaded_by != context.user_id):
                raise HTTPException(404, "Active document not found.")
            document.active, document.deleted_at = False, datetime.now(timezone.utc)
            session.execute(update(DocumentVersion).where(DocumentVersion.document_id == document_id).values(active=False))
            session.execute(update(Chunk).where(Chunk.document_id == document_id).values(active=False))
        _audit(context, "document.delete", "document", document_id)
        return {"document_id": document_id, "status": "deleted", "purge_after_days": SETTINGS.deleted_retention_days}
    registry, service, _, _ = _runtime(context)
    document = next((row for row in registry.list_documents() if row["document_id"] == document_id and row["active"]), None)
    if not document or (context.role == "editor" and document.get("uploaded_by") != context.user_id):
        raise HTTPException(404, "Active document not found.")
    if not service.delete_document(document_id):
        raise HTTPException(404, "Active document not found.")
    tenant_stores.invalidate(context.organization_id)
    _audit(context, "document.delete", "document", document_id)
    return {"document_id": document_id, "status": "deleted"}


@app.get("/reviews")
def reviews(status: Literal["open", "resolved", "all"] = "open", limit: int = 100, context: SecurityContext = Depends(context_from_bearer)):
    _require(context, "reviews:write")
    registry = PostgresReviewRegistry(platform_runtime.database, context) if platform_runtime else _runtime(context)[3]
    return {"cases": registry.list_cases(status, min(max(limit, 1), 500))}


@app.get("/reviews/{case_id}")
def review(case_id: str, context: SecurityContext = Depends(context_from_bearer)):
    _require(context, "reviews:write")
    registry = PostgresReviewRegistry(platform_runtime.database, context) if platform_runtime else _runtime(context)[3]
    value = registry.get_case(case_id)
    if not value:
        raise HTTPException(404, "Review case not found.")
    return value


@app.post("/reviews/{case_id}/decision")
def decide(case_id: str, body: ReviewDecisionRequest, context: SecurityContext = Depends(context_from_bearer)):
    _require(context, "reviews:write")
    registry = PostgresReviewRegistry(platform_runtime.database, context) if platform_runtime else _runtime(context)[3]
    if not registry.decide(case_id, body.decision, context.user_id, body.notes):
        raise HTTPException(404, "Review case not found.")
    _audit(context, "review.decide", "review_case", case_id, details={"decision": body.decision})
    return registry.get_case(case_id)


def list_review_cases(status="open", limit=100, context=None):
    if not isinstance(context, SecurityContext):
        return {"cases": review_registry.list_cases(status, limit)}
    return reviews(status, limit, context)


def get_review_case(case_id: str, context=None):
    if not isinstance(context, SecurityContext):
        value = review_registry.get_case(case_id)
        if not value:
            raise HTTPException(404, "Review case not found.")
        return value
    return review(case_id, context)


def submit_review_decision(case_id: str, body: ReviewDecisionRequest, context=None):
    if not isinstance(context, SecurityContext):
        if not review_registry.decide(case_id, body.decision, body.reviewer, body.notes):
            raise HTTPException(404, "Review case not found.")
        return review_registry.get_case(case_id)
    return decide(case_id, body, context)


@app.post("/api-keys", status_code=201)
def create_key(body: ApiKeyCreateRequest, context: SecurityContext = Depends(context_from_bearer)):
    return security_registry.create_key_for_context(context, body.scopes, body.expires_days)


@app.get("/api-keys")
def keys(context: SecurityContext = Depends(context_from_bearer)):
    return {"api_keys": security_registry.list_api_keys(context)}


@app.delete("/api-keys/{key_id}")
def revoke_key(key_id: str, context: SecurityContext = Depends(context_from_bearer)):
    try:
        security_registry.revoke_api_key(context, key_id)
    except KeyError as exc:
        raise HTTPException(404, "API key not found.") from exc
    return {"key_id": key_id, "status": "revoked"}


@app.get("/members")
def members(context: SecurityContext = Depends(context_from_bearer)):
    return {"members": security_registry.list_members(context)}


@app.post("/members", status_code=201)
def add_member(body: MembershipCreateRequest, context: SecurityContext = Depends(context_from_bearer)):
    return {"user_id": security_registry.add_user(context, body.email, body.display_name, body.role)}


@app.patch("/members/{user_id}")
def update_member(user_id: str, body: MembershipUpdateRequest, context: SecurityContext = Depends(context_from_bearer)):
    try:
        security_registry.change_membership(context, user_id, body.role)
    except KeyError as exc:
        raise HTTPException(404, "Membership not found.") from exc
    return {"user_id": user_id, "role": body.role}


@app.post("/members/{user_id}/oidc", status_code=201)
def bind_oidc_identity(user_id: str, body: OidcIdentityRequest, context: SecurityContext = Depends(context_from_bearer)):
    if not platform_runtime or not isinstance(security_registry, PostgresSecurityRegistry):
        raise HTTPException(409, "OIDC identity binding requires PostgreSQL platform mode.")
    if body.issuer.rstrip("/") != SETTINGS.oidc_issuer.rstrip("/"):
        raise HTTPException(422, "OIDC issuer does not match the configured provider.")
    try:
        security_registry.register_oidc_identity(context, user_id, body.issuer, body.subject)
    except KeyError as exc:
        raise HTTPException(404, "Membership not found.") from exc
    _audit(context, "member.oidc.bind", "membership", user_id)
    return {"user_id": user_id, "issuer": body.issuer, "status": "bound"}


@app.get("/quarantine")
def quarantine(status: str | None = None, context: SecurityContext = Depends(context_from_bearer)):
    return {"cases": security_registry.list_quarantine(context, status)}


@app.get("/quarantine/{case_id}")
def quarantine_detail(case_id: str, context: SecurityContext = Depends(context_from_bearer)):
    value = security_registry.get_quarantine_internal(context, case_id)
    if not value:
        raise HTTPException(404, "Quarantine case not found.")
    value.pop("staged_path", None)
    return value


def _resolve_quarantine(case_id: str, body: QuarantineDecisionRequest, context: SecurityContext, decision: str):
    case = security_registry.get_quarantine_internal(context, case_id)
    if not case:
        raise HTTPException(404, "Quarantine case not found.")
    try:
        security_registry.resolve_quarantine(context, case_id, decision, body.reason)
    except KeyError as exc:
        raise HTTPException(404, "Quarantine case not found.") from exc
    job_id = None
    if decision == "approved":
        if platform_runtime:
            from dataclasses import asdict
            from backend.database import GlobalJobLookup, IngestionJob
            content = platform_runtime.object_store.get(context.organization_id, case["staged_path"])
            job_id = "job_" + uuid.uuid4().hex
            options = IngestionOptions(embedding_model=SETTINGS.embedding_model)
            with platform_runtime.database.session(context, write=True) as session:
                session.add(IngestionJob(
                    organization_id=context.organization_id, job_id=job_id, created_by=context.user_id,
                    state="queued", payload_json={"object_key": case["staged_path"], "source_name": case["document_name"],
                    "content_type": "application/octet-stream", "checksum_sha256": hashlib.sha256(content).hexdigest(),
                    "options": asdict(options)}, progress_json={"stage": "queued", "progress": 0},
                ))
                session.add(GlobalJobLookup(job_id=job_id, organization_id=context.organization_id, created_by=context.user_id))
            platform_runtime.queue.enqueue(job_id)
            return {"case_id": case_id, "status": decision, "job_id": job_id}
        registry, _, worker, _ = _runtime(context)
        options = IngestionOptions(embedding_model=SETTINGS.embedding_model)
        job_id = registry.enqueue_job({
            "path": case["staged_path"], "source_name": case["document_name"], "options": options.__dict__,
            "organization_id": context.organization_id, "created_by": context.user_id,
            "trust_state": "trusted", "security_release_case_id": case_id,
        }, created_by=context.user_id)
        worker.start()
    return {"case_id": case_id, "status": decision, "job_id": job_id}


@app.post("/quarantine/{case_id}/approve")
def approve(case_id: str, body: QuarantineDecisionRequest, context: SecurityContext = Depends(context_from_bearer)):
    return _resolve_quarantine(case_id, body, context, "approved")


@app.post("/quarantine/{case_id}/reject")
def reject(case_id: str, body: QuarantineDecisionRequest, context: SecurityContext = Depends(context_from_bearer)):
    return _resolve_quarantine(case_id, body, context, "rejected")


@app.get("/audit")
def audit(limit: int = 100, context: SecurityContext = Depends(context_from_bearer)):
    event = _audit(context, "audit.read", "audit")
    return {"access_event_id": event, "events": security_registry.list_audit(context, limit)}


@app.get("/audit/verify")
def verify_audit(context: SecurityContext = Depends(context_from_bearer)):
    _require(context, "security:read")
    value = security_registry.verify_audit_chain(context) if platform_runtime else security_registry.verify_audit_chain()
    value["access_event_id"] = _audit(context, "audit.verify", "audit")
    return value


@app.get("/audit/export")
def export_audit(context: SecurityContext = Depends(context_from_bearer)):
    rows = security_registry.list_audit(context, 1000)
    return {"access_event_id": _audit(context, "audit.export", "audit"), "format": "jsonl",
            "content": "\n".join(json.dumps(row, sort_keys=True, default=str) for row in reversed(rows))}


@app.get("/health")
def health():
    if platform_runtime:
        platform = platform_runtime.health()
        return {"status": "ok" if platform["ready"] else "degraded", "platform": platform,
                "security": {"mode": SETTINGS.security_mode, "ready": security_registry.protected_ready,
                             "isolation": "postgres_rls", "authentication": ["oidc", "api_key"]},
                "grounding": {"enabled": True, "policy": "strict"}}
    public = _store(_anonymous())
    verifier, policy = get_default_verifier(), default_grounding_policy()
    ready = SETTINGS.security_mode == "demo" or security_registry.protected_ready
    try:
        with security_registry._connect() as connection:
            quarantined = connection.execute("SELECT COUNT(*) FROM quarantine_cases WHERE status='open'").fetchone()[0]
    except Exception:
        quarantined = 0
    return {"status": "ok" if ready else "degraded", "index_loaded": public.index is not None,
            "security": {"mode": SETTINGS.security_mode, "ready": ready, "isolation": "per_organization_faiss",
                         "open_quarantine_cases": quarantined, "scanner": scanner_capabilities()},
            "ingestion_capabilities": parser_capabilities(),
            "grounding": {"enabled": True, "model": verifier.model_name, "revision": verifier.model_revision,
                          "loaded": verifier.is_loaded, "policy": "strict", "policy_id": policy.policy_id,
                          "premise_version": policy.premise_version, "guard_version": policy.guard_version}}


@app.get("/ready")
def ready():
    value = health()
    if value["status"] != "ok":
        raise HTTPException(503, value)
    return value
