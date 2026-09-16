"""Context-aware ingestion orchestration, atomic FAISS activation, and worker."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import SETTINGS
from .observability import span
from .contextual_chunking import CHUNKER_VERSION, CONTEXT_PROMPT_VERSION, build_contextual_chunks
from .document_parsers import PARSER_VERSION, DocumentParseError, SUPPORTED_SUFFIXES, parse_document, source_sha256, stable_document_id
from .experiments import corpus_fingerprint
from .ingestion_registry import IngestionRegistry
from .utils import ensure_dir, write_json
from .vectorstore import FaissStore
from .security import SecurityRegistry
from .security_scanner import scan_upload

MANIFEST_FILE = "manifest.json"
MANIFEST_SCHEMA_VERSION = "2.0"


class QuarantinedUpload(DocumentParseError):
    def __init__(self, case_id: str):
        super().__init__("QUARANTINED", "The upload was quarantined for security review.")
        self.case_id = case_id


@dataclass(frozen=True)
class IngestionOptions:
    child_tokens: int = 420
    overlap_tokens: int = 80
    parent_tokens: int = 1200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    context_mode: str = "deterministic"
    context_model: str = "gemini-2.5-flash"
    context_prompt_version: str = "1.0"
    source_version: str | None = None


def parser_capabilities() -> Dict[str, bool]:
    import importlib.util
    return {
        "pdf": importlib.util.find_spec("fitz") is not None,
        "tables": importlib.util.find_spec("pdfplumber") is not None,
        "docx": importlib.util.find_spec("docx") is not None,
        "html": importlib.util.find_spec("bs4") is not None,
        "ocr": importlib.util.find_spec("pytesseract") is not None and bool(shutil.which(SETTINGS.ocr_executable or "tesseract")),
    }


def dependency_versions() -> Dict[str, str]:
    result = {}
    for package in ("PyMuPDF", "pdfplumber", "python-docx", "beautifulsoup4", "pytesseract", "faiss-cpu", "sentence-transformers"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def index_manifest(items: List[Dict[str, Any]], options: IngestionOptions, active_documents: List[Dict[str, Any]], organization_id: str = "org_public") -> Dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "context_prompt_version": options.context_prompt_version,
        "context_mode": options.context_mode,
        "context_model": options.context_model if options.context_mode == "gemini" else None,
        "embedding_model": options.embedding_model,
        "child_tokens": options.child_tokens,
        "overlap_tokens": options.overlap_tokens,
        "parent_tokens": options.parent_tokens,
        "corpus_fingerprint": corpus_fingerprint(items),
        "built_at": time.time(),
        "active_document_versions": sorted(
            [document["current_version_id"] for document in active_documents if document.get("active") and document.get("current_version_id")]
        ),
        "capabilities": parser_capabilities(),
        "dependency_versions": dependency_versions(),
        "organization_id": organization_id,
        "isolation_capability": "per_organization_faiss",
        "security_policy_version": SETTINGS.security_policy_version,
        "authorized_document_versions": sorted(
            [document["current_version_id"] for document in active_documents if document.get("active") and document.get("current_version_id")]
        ),
    }


def activate_faiss_atomically(index_dir: str, vectors: np.ndarray, items: List[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    target = Path(index_dir).resolve()
    ensure_dir(str(target.parent))
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-staging-", dir=str(target.parent)))
    backup = target.parent / f".{target.name}-backup-{os.getpid()}-{int(time.time() * 1000)}"
    moved_old = False
    try:
        if not items or vectors.ndim != 2 or len(items) != vectors.shape[0]:
            raise ValueError("Every active chunk must align with exactly one vector.")
        staged_store = FaissStore(str(staging))
        staged_store.build(np.asarray(vectors, dtype="float32"), items)
        write_json(str(staging / MANIFEST_FILE), manifest)
        validator = FaissStore(str(staging))
        if not validator.load() or validator.index.ntotal != len(items) or len(validator.meta.get("items", [])) != len(items):
            raise RuntimeError("Staged FAISS index failed alignment validation.")
        if target.exists():
            os.replace(target, backup)
            moved_old = True
        os.replace(staging, target)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if moved_old and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


class IngestionService:
    def __init__(self, registry: IngestionRegistry, index_dir: str, upload_dir: str, embedder_factory,
                 gemini_client=None, organization_id: str = "org_public", actor_user_id: str = "",
                 security_registry: SecurityRegistry | None = None):
        self.registry = registry
        self.index_dir = index_dir
        self.upload_dir = upload_dir
        self.embedder_factory = embedder_factory
        self.gemini_client = gemini_client
        self.organization_id = organization_id
        self.actor_user_id = actor_user_id
        self.security_registry = security_registry
        self._embedders: Dict[str, Any] = {}
        ensure_dir(upload_dir)

    def _embedder(self, model: str):
        if model not in self._embedders:
            self._embedders[model] = self.embedder_factory(model)
        return self._embedders[model]

    def _validate_index_options(self, options: IngestionOptions) -> None:
        manifest_path = Path(self.index_dir) / MANIFEST_FILE
        if not manifest_path.exists() or not any(document.get("active") for document in self.registry.list_documents()):
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        expected = {
            "child_tokens": options.child_tokens,
            "overlap_tokens": options.overlap_tokens,
            "parent_tokens": options.parent_tokens,
            "context_mode": options.context_mode,
            "context_prompt_version": options.context_prompt_version,
        }
        if options.context_mode == "gemini":
            expected["context_model"] = options.context_model
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            raise DocumentParseError(
                "INDEX_CONFIG_MISMATCH",
                "Index-wide settings differ for: " + ", ".join(mismatches) + ". Delete/rebuild the managed corpus to change them.",
            )

    def stage_upload(self, filename: str, content: bytes, options: IngestionOptions, declared_mime: str | None = None) -> str:
        if len(content) > SETTINGS.max_upload_mb * 1024 * 1024:
            raise DocumentParseError("FILE_TOO_LARGE", f"Files must be no larger than {SETTINGS.max_upload_mb} MB.")
        safe_name = Path(filename).name
        suffix = Path(safe_name).suffix.casefold()
        if suffix not in SUPPORTED_SUFFIXES:
            raise DocumentParseError("UNSUPPORTED_FORMAT", f"Unsupported document type: {suffix}")
        if suffix == ".pdf" and not content.startswith(b"%PDF"):
            raise DocumentParseError("INVALID_FILE_SIGNATURE", "The file extension is PDF but the content is not a PDF.")
        if suffix == ".docx" and not content.startswith(b"PK\x03\x04"):
            raise DocumentParseError("INVALID_FILE_SIGNATURE", "The file extension is DOCX but the content is not an Office document.")
        report = scan_upload(safe_name, content, declared_mime)
        checksum = hashlib.sha256(content).hexdigest()
        staged_dir = Path(self.upload_dir) / self.organization_id / uuid.uuid4().hex
        ensure_dir(str(staged_dir))
        path = staged_dir / (uuid.uuid4().hex + suffix)
        path.write_bytes(content)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        if report.quarantined:
            if not self.security_registry:
                raise DocumentParseError("UNSAFE_UPLOAD", "The upload failed security validation.")
            case_id = self.security_registry.create_quarantine_case(
                self.organization_id, safe_name, [finding.to_dict() for finding in report.findings],
                self.actor_user_id, str(path),
            )
            self.security_registry.append_audit(
                self.organization_id, self.actor_user_id, None, "ingestion.quarantined", "quarantine_case",
                case_id, "blocked", 202, "SECURITY_FINDING", details={"finding_codes": [finding.code for finding in report.findings]},
            )
            raise QuarantinedUpload(case_id)
        trust_state = "public" if self.organization_id == "org_public" else ("trusted" if self.actor_user_id.startswith("admin:") else "untrusted")
        return self.registry.enqueue_job({
            "path": str(path), "source_name": safe_name, "options": asdict(options),
            "organization_id": self.organization_id, "created_by": self.actor_user_id,
            "trust_state": trust_state, "security_findings": [finding.to_dict() for finding in report.findings],
            "source_sha256": checksum,
        }, created_by=self.actor_user_id)

    def _context_enhancer(self, options: IngestionOptions):
        if options.context_mode != "gemini" or self.gemini_client is None:
            return None

        def enhance(prefix: str, text: str) -> str | None:
            key = hashlib.sha256(f"{prefix}\x1f{text}\x1f{options.context_model}\x1f{options.context_prompt_version}".encode("utf-8")).hexdigest()
            cached = self.registry.get_context(key)
            if cached:
                return cached
            prompt = (
                "Write one factual context sentence, at most 60 words, that locates this chunk within its document. "
                "Do not add facts not present in the metadata or chunk.\n"
                f"Metadata: {prefix}\nChunk: {text}"
            )
            try:
                response = self.gemini_client.models.generate_content(model=options.context_model, contents=prompt)
                value = " ".join((response.text or "").split())
                value = " ".join(value.split()[:60])
            except Exception:
                return None
            if value:
                self.registry.put_context(key, options.context_model, options.context_prompt_version, value)
            return value or None
        return enhance

    def process_payload(self, payload: Dict[str, Any], progress=lambda stage, value, message: None) -> Dict[str, Any]:
        path, source_name = payload["path"], payload["source_name"]
        options = IngestionOptions(**payload.get("options", {}))
        progress("validating", 0.05, "Validating file")
        self._validate_index_options(options)
        checksum = source_sha256(path)
        if payload.get("organization_id", self.organization_id) != self.organization_id:
            raise RuntimeError("Tenant mismatch in ingestion job payload.")
        document_id = stable_document_id(source_name, self.organization_id)
        previous_checksum = self.registry.current_checksum(document_id)
        if previous_checksum == checksum:
            manifest_path = Path(self.index_dir) / MANIFEST_FILE
            manifest_versions = set()
            manifest_embedding_model = None
            if manifest_path.exists():
                try:
                    current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest_versions = set(current_manifest.get("active_document_versions", []))
                    manifest_embedding_model = current_manifest.get("embedding_model")
                except (OSError, json.JSONDecodeError):
                    pass
            registry_versions = {
                document["current_version_id"] for document in self.registry.list_documents()
                if document.get("active") and document.get("current_version_id")
            }
            if manifest_versions == registry_versions and "ver_" + checksum[:20] in manifest_versions and manifest_embedding_model == options.embedding_model:
                return {"outcome": "unchanged", "document_id": document_id, "source_sha256": checksum, "warnings": []}
        progress("parsing", 0.15, "Parsing structured content")
        with span("rag.ingestion.parse", {"rag.document.mime_known": bool(source_name)}):
            parsed = parse_document(
                path, source_name=source_name, source_version=options.source_version,
                ocr_executable=SETTINGS.ocr_executable or None, organization_id=self.organization_id,
                created_by=payload.get("created_by", self.actor_user_id), trust_state=payload.get("trust_state", "untrusted"),
            )
        progress("ocr", 0.28, "OCR and table extraction checks completed")
        progress("chunking", 0.35, "Building parent and child chunks")
        progress("context", 0.45, f"Applying {options.context_mode} chunk context")
        with span("rag.ingestion.chunk", {"rag.chunk.target_tokens": options.child_tokens}):
            chunks = build_contextual_chunks(
                parsed.document, parsed.version, parsed.blocks,
                child_tokens=options.child_tokens, overlap_tokens=options.overlap_tokens,
                parent_tokens=options.parent_tokens, context_enhancer=self._context_enhancer(options),
            )
        if not chunks:
            raise DocumentParseError("EMPTY_DOCUMENT", "Parsing produced no searchable chunks.")
        progress("embedding", 0.55, "Embedding new contextual chunks")
        missing = [chunk for chunk in chunks if self.registry.get_cached_embedding(chunk.content_fingerprint, options.embedding_model) is None]
        if missing:
            embedder = self._embedder(options.embedding_model)
            with span("rag.ingestion.embed", {"rag.embedding.count": len(missing), "rag.embedding.model": options.embedding_model}):
                new_vectors = embedder.embed_texts([chunk.retrieval_text for chunk in missing])
            self.registry.put_cached_embeddings(options.embedding_model, missing, new_vectors)
        lock_owner = uuid.uuid4().hex
        if not self.registry.acquire_lock("index_activation", lock_owner):
            raise RuntimeError("Timed out waiting for the index activation lock.")
        previous_state = None
        try:
            previous_state = self.registry.document_state(parsed.document.document_id)
            self.registry.activate_version(parsed.document, parsed.version, chunks, path)
            progress("index_build", 0.78, "Building active FAISS generation")
            active_payloads = self.registry.active_chunk_payloads()
            missing_active = [
                item for item in active_payloads
                if self.registry.get_cached_embedding(item["content_fingerprint"], options.embedding_model) is None
            ]
            if missing_active:
                embedder = self._embedder(options.embedding_model)
                vectors_for_active = embedder.embed_texts([item["retrieval_text"] for item in missing_active])
                self.registry.put_cached_payload_embeddings(options.embedding_model, missing_active, vectors_for_active)
            items, vectors = self.registry.active_items_and_vectors(options.embedding_model)
            documents = self.registry.list_documents()
            if any(item.get("organization_id") != self.organization_id for item in items):
                raise RuntimeError("Cross-tenant chunk detected before index activation.")
            manifest = index_manifest(items, options, documents, self.organization_id)
            progress("validation", 0.88, "Validating vector and metadata alignment")
            progress("activation", 0.94, "Activating the staged index generation")
            try:
                activate_faiss_atomically(self.index_dir, vectors, items, manifest)
            except Exception:
                self.registry.restore_document_state(parsed.document.document_id, previous_state)
                raise
        finally:
            self.registry.release_lock("index_activation", lock_owner)
        return {
            "outcome": "updated" if previous_checksum else "indexed",
            "document_id": parsed.document.document_id,
            "document_version_id": parsed.version.document_version_id,
            "chunk_count": len(chunks),
            "corpus_fingerprint": manifest["corpus_fingerprint"],
            "warnings": parsed.warnings,
        }

    def delete_document(self, document_id: str, embedding_model: str | None = None) -> bool:
        lock_owner = uuid.uuid4().hex
        if not self.registry.acquire_lock("index_activation", lock_owner):
            raise RuntimeError("Timed out waiting for the index activation lock.")
        previous_state = self.registry.document_state(document_id)
        try:
            if not self.registry.soft_delete_document(document_id):
                return False
            model = embedding_model or SETTINGS.embedding_model
            items, vectors = self.registry.active_items_and_vectors(model)
            if items:
                options = IngestionOptions(embedding_model=model)
                activate_faiss_atomically(self.index_dir, vectors, items, index_manifest(items, options, self.registry.list_documents()))
            else:
                target = Path(self.index_dir)
                if target.exists():
                    backup = target.parent / f".{target.name}-empty-{int(time.time() * 1000)}"
                    os.replace(target, backup)
                    shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            self.registry.restore_document_state(document_id, previous_state)
            raise
        finally:
            self.registry.release_lock("index_activation", lock_owner)
        return True


class IngestionWorker:
    def __init__(self, service: IngestionService, lease_seconds: int = 120, poll_seconds: float = 0.5):
        self.service, self.lease_seconds, self.poll_seconds = service, lease_seconds, poll_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.service.registry.recover_stale_jobs()
        self._thread = threading.Thread(target=self._run, name="ingestion-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self.service.registry.claim_job(self.lease_seconds)
            if not job:
                self._stop.wait(self.poll_seconds)
                continue
            job_id = job["job_id"]
            try:
                result = self.service.process_payload(
                    job["payload"],
                    lambda stage, value, message: self.service.registry.update_job(job_id, stage, value, message, self.lease_seconds),
                )
                self.service.registry.finish_job(job_id, result, result.pop("warnings", []))
            except DocumentParseError as exc:
                self.service.registry.fail_job(job_id, exc.code, str(exc))
            except Exception as exc:
                self.service.registry.fail_job(job_id, "INGESTION_FAILED", str(exc))
