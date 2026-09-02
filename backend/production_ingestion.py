"""Distributed, PostgreSQL-authoritative ingestion coordinator."""
from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from .config import SETTINGS
from .contextual_chunking import build_contextual_chunks
from .database import DatabaseRuntime, Document, GlobalJobLookup, IngestionJob, JobEvent, ObjectRecord, QuarantineCase
from .document_parsers import DocumentParseError, parse_document
from .embeddings import Embedder
from .ingestion import IngestionOptions
from .object_store import S3EnvelopeObjectStore
from .postgres_store import PgVectorStore
from .security_models import SecurityContext
from .security_scanner import scan_upload


class ProductionIngestionService:
    def __init__(self, database: DatabaseRuntime, object_store, queue, embedder_factory=Embedder):
        self.database = database
        self.object_store = object_store
        self.queue = queue
        self.embedder_factory = embedder_factory
        self._embedders: dict[str, Any] = {}

    def _embedder(self, model_name: str):
        if model_name not in self._embedders:
            self._embedders[model_name] = self.embedder_factory(model_name)
        return self._embedders[model_name]

    def enqueue_upload(
        self,
        context: SecurityContext,
        filename: str,
        content: bytes,
        content_type: str,
        options: IngestionOptions,
    ) -> dict[str, Any]:
        context.require("documents:write")
        if context.organization_id == SETTINGS.public_organization_id:
            raise PermissionError("The public organization cannot accept uploads.")
        if len(content) > SETTINGS.max_upload_mb * 1024 * 1024:
            raise DocumentParseError("FILE_TOO_LARGE", "The document exceeds the configured upload limit.")
        with self.database.session(context) as session:
            document_count = session.scalar(select(func.count()).select_from(Document).where(Document.active.is_(True))) or 0
            concurrent = session.scalar(
                select(func.count()).select_from(IngestionJob).where(IngestionJob.state.in_(("queued", "running")))
            ) or 0
        if document_count >= SETTINGS.quota_documents:
            raise DocumentParseError("DOCUMENT_QUOTA_EXCEEDED", "The organization document quota is exhausted.")
        if concurrent >= SETTINGS.quota_concurrent_jobs:
            raise DocumentParseError("CONCURRENT_JOB_QUOTA_EXCEEDED", "The organization has too many active ingestion jobs.")
        report = scan_upload(filename, content, content_type)
        stored = self.object_store.put(
            context.organization_id,
            filename,
            content,
            content_type,
            {"uploader": context.user_id, "trust": "quarantined" if report.quarantined else "untrusted"},
        )
        if report.quarantined:
            case_id = "qua_" + uuid.uuid4().hex
            with self.database.session(context, write=True) as session:
                session.add(ObjectRecord(
                    organization_id=context.organization_id, object_key=stored.object_key, version_id=stored.version_id,
                    checksum_sha256=stored.checksum_sha256, size_bytes=stored.size_bytes, content_type=content_type,
                    encryption_algorithm=stored.encryption_algorithm,
                ))
                session.add(QuarantineCase(
                    organization_id=context.organization_id,
                    case_id=case_id,
                    document_name=Path(filename).name,
                    object_key=stored.object_key,
                    findings_json=[finding.to_dict() for finding in report.findings],
                    created_by=context.user_id,
                ))
            return {"status": "quarantined", "case_id": case_id, "findings": [finding.to_dict() for finding in report.findings]}
        job_id = "job_" + uuid.uuid4().hex
        payload = {
            "object_key": stored.object_key,
            "source_name": Path(filename).name,
            "content_type": content_type,
            "checksum_sha256": stored.checksum_sha256,
            "options": asdict(options),
        }
        with self.database.session(context, write=True) as session:
            session.add(ObjectRecord(
                organization_id=context.organization_id, object_key=stored.object_key, version_id=stored.version_id,
                checksum_sha256=stored.checksum_sha256, size_bytes=stored.size_bytes, content_type=content_type,
                encryption_algorithm=stored.encryption_algorithm,
            ))
            session.add(IngestionJob(
                organization_id=context.organization_id,
                job_id=job_id,
                created_by=context.user_id,
                state="queued",
                payload_json=payload,
                progress_json={"stage": "queued", "progress": 0},
            ))
            session.add(GlobalJobLookup(job_id=job_id, organization_id=context.organization_id, created_by=context.user_id))
            self._event(session, context.organization_id, job_id, "queued", 0, "Upload validated and queued")
        self.queue.enqueue(job_id)
        return {"status": "queued", "job_id": job_id, "checksum_sha256": stored.checksum_sha256}

    def process(self, context: SecurityContext, job_id: str) -> dict[str, Any]:
        with self.database.session(context, write=True) as session:
            job = session.execute(
                select(IngestionJob).where(IngestionJob.job_id == job_id).with_for_update(skip_locked=True)
            ).scalar_one_or_none()
            if job is None:
                raise KeyError("Ingestion job not found.")
            if job.state in {"succeeded", "succeeded_with_warnings", "cancelled"}:
                return dict(job.progress_json)
            if job.state == "running":
                return {"status": "already_running"}
            job.state = "running"
            job.attempts += 1
            job.updated_at = datetime.now(timezone.utc)
            self._event(session, context.organization_id, job_id, "running", 5, "Worker lease acquired")
            payload = dict(job.payload_json)
        try:
            content = self.object_store.get(context.organization_id, payload["object_key"])
            if hashlib.sha256(content).hexdigest() != payload["checksum_sha256"]:
                raise RuntimeError("Stored upload checksum mismatch.")
            suffix = Path(payload["source_name"]).suffix.casefold()
            temporary = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            try:
                temporary.write(content)
                temporary.close()
                options = IngestionOptions(**payload["options"])
                parsed = parse_document(
                    temporary.name,
                    source_name=payload["source_name"],
                    source_version=options.source_version,
                    ocr_executable=SETTINGS.ocr_executable or None,
                    organization_id=context.organization_id,
                    created_by=context.user_id,
                    trust_state="untrusted",
                )
                chunks = build_contextual_chunks(
                    parsed.document,
                    parsed.version,
                    parsed.blocks,
                    child_tokens=options.child_tokens,
                    overlap_tokens=options.overlap_tokens,
                    parent_tokens=options.parent_tokens,
                )
                items = []
                for chunk in chunks:
                    item = chunk.to_dict()
                    item["embedding_model"] = options.embedding_model
                    items.append(item)
                vectors = self._embedder(options.embedding_model).embed_texts([item["retrieval_text"] for item in items])
                outcome = PgVectorStore(self.database, context).upsert_document_chunks(
                    parsed.document, parsed.version, items, vectors, payload["object_key"]
                )
                state = "succeeded_with_warnings" if parsed.warnings else "succeeded"
                result = {
                    "status": state,
                    "outcome": outcome,
                    "document_id": parsed.document.document_id,
                    "document_version_id": parsed.version.document_version_id,
                    "chunk_count": len(items),
                    "warnings": parsed.warnings,
                }
            finally:
                try:
                    os.unlink(temporary.name)
                except OSError:
                    pass
        except Exception as exc:
            with self.database.session(context, write=True) as session:
                job = session.execute(select(IngestionJob).where(IngestionJob.job_id == job_id).with_for_update()).scalar_one()
                job.state = "failed"
                job.error_code = getattr(exc, "code", "INGESTION_FAILED")
                job.progress_json = {"stage": "failed", "progress": 100, "error_code": job.error_code}
                job.updated_at = datetime.now(timezone.utc)
                self._event(session, context.organization_id, job_id, "failed", 100, job.error_code)
            raise
        with self.database.session(context, write=True) as session:
            job = session.execute(select(IngestionJob).where(IngestionJob.job_id == job_id).with_for_update()).scalar_one()
            job.state = result["status"]
            job.progress_json = result
            job.updated_at = datetime.now(timezone.utc)
            self._event(session, context.organization_id, job_id, result["status"], 100, "Index update committed")
        return result

    @staticmethod
    def _event(session, organization_id: str, job_id: str, stage: str, progress: int, message: str) -> None:
        session.add(JobEvent(
            organization_id=organization_id,
            event_id="jev_" + uuid.uuid4().hex,
            job_id=job_id,
            stage=stage,
            progress=progress,
            message=message[:512],
        ))
