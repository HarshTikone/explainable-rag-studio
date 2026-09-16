"""SQLite-backed document lifecycle, job journal, caches, and active chunks."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .ingestion_models import ChildChunk, Document, DocumentVersion
from .utils import ensure_dir


def _now() -> float:
    return time.time()


class IngestionRegistry:
    def __init__(self, path: str, organization_id: str = "org_public"):
        self.path = path
        self.organization_id = organization_id
        ensure_dir(str(__import__("pathlib").Path(path).parent))
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def transaction(self):
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY, logical_source TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL, mime_type TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
                    current_version_id TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_versions (
                    document_version_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL, source_version TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                    source_path TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL,
                    UNIQUE(document_id, source_sha256), FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, document_version_id TEXT NOT NULL,
                    content_fingerprint TEXT NOT NULL, payload_json TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(active, document_id);
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    content_fingerprint TEXT NOT NULL, embedding_model TEXT NOT NULL,
                    dimension INTEGER NOT NULL, vector BLOB NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(content_fingerprint, embedding_model)
                );
                CREATE TABLE IF NOT EXISTS context_cache (
                    cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
                    context TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL, progress REAL NOT NULL,
                    payload_json TEXT NOT NULL, result_json TEXT, warnings_json TEXT NOT NULL DEFAULT '[]',
                    error_code TEXT, error_message TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                    lease_until REAL, created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON ingestion_jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS job_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, stage TEXT NOT NULL,
                    progress REAL NOT NULL, message TEXT NOT NULL, created_at REAL NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES ingestion_jobs(job_id)
                );
                CREATE TABLE IF NOT EXISTS lifecycle_locks (
                    name TEXT PRIMARY KEY, owner TEXT NOT NULL, lease_until REAL NOT NULL
                );
            """)
            self._ensure_column(connection, "documents", "organization_id", "TEXT NOT NULL DEFAULT 'org_public'")
            self._ensure_column(connection, "documents", "created_by", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, "documents", "uploaded_by", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(connection, "documents", "trust_state", "TEXT NOT NULL DEFAULT 'public'")
            self._ensure_column(connection, "document_versions", "organization_id", "TEXT NOT NULL DEFAULT 'org_public'")
            self._ensure_column(connection, "chunks", "organization_id", "TEXT NOT NULL DEFAULT 'org_public'")
            self._ensure_column(connection, "ingestion_jobs", "organization_id", "TEXT NOT NULL DEFAULT 'org_public'")
            self._ensure_column(connection, "ingestion_jobs", "created_by", "TEXT NOT NULL DEFAULT ''")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_org ON documents(organization_id, active)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_org ON chunks(organization_id, active)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_jobs_org ON ingestion_jobs(organization_id, status, created_at)")

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, name: str, declaration: str) -> None:
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if name not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    def acquire_lock(self, name: str, owner: str, lease_seconds: int = 300, timeout_seconds: float = 30.0) -> bool:
        deadline = _now() + timeout_seconds
        while _now() < deadline:
            with self.transaction() as connection:
                row = connection.execute("SELECT owner,lease_until FROM lifecycle_locks WHERE name=?", (name,)).fetchone()
                if not row or row["lease_until"] < _now() or row["owner"] == owner:
                    connection.execute(
                        "INSERT INTO lifecycle_locks VALUES (?, ?, ?) ON CONFLICT(name) DO UPDATE SET owner=excluded.owner,lease_until=excluded.lease_until",
                        (name, owner, _now() + lease_seconds),
                    )
                    return True
            time.sleep(0.05)
        return False

    def release_lock(self, name: str, owner: str) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM lifecycle_locks WHERE name=? AND owner=?", (name, owner))

    def current_checksum(self, document_id: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT v.source_sha256 FROM documents d JOIN document_versions v ON v.document_version_id=d.current_version_id WHERE d.document_id=? AND d.organization_id=? AND d.active=1",
                (document_id, self.organization_id),
            ).fetchone()
        return row[0] if row else None

    def get_cached_embedding(self, fingerprint: str, model: str) -> np.ndarray | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT dimension, vector FROM embedding_cache WHERE content_fingerprint=? AND embedding_model=?",
                (fingerprint, model),
            ).fetchone()
        from .telemetry import record_cache_event
        record_cache_event("embedding", bool(row))
        return np.frombuffer(row["vector"], dtype="float32").copy() if row else None

    def put_cached_embeddings(self, model: str, chunks: Sequence[ChildChunk], vectors: np.ndarray) -> None:
        with self.transaction() as connection:
            for chunk, vector in zip(chunks, vectors):
                value = np.asarray(vector, dtype="float32").reshape(-1)
                connection.execute(
                    "INSERT OR REPLACE INTO embedding_cache VALUES (?, ?, ?, ?, ?)",
                    (chunk.content_fingerprint, model, len(value), value.tobytes(), _now()),
                )

    def put_cached_payload_embeddings(self, model: str, items: Sequence[Dict[str, Any]], vectors: np.ndarray) -> None:
        with self.transaction() as connection:
            for item, vector in zip(items, vectors):
                value = np.asarray(vector, dtype="float32").reshape(-1)
                connection.execute(
                    "INSERT OR REPLACE INTO embedding_cache VALUES (?, ?, ?, ?, ?)",
                    (item["content_fingerprint"], model, len(value), value.tobytes(), _now()),
                )

    def get_context(self, cache_key: str) -> str | None:
        with self.connect() as connection:
            row = connection.execute("SELECT context FROM context_cache WHERE cache_key=?", (cache_key,)).fetchone()
        from .telemetry import record_cache_event
        record_cache_event("context", bool(row))
        return row[0] if row else None

    def put_context(self, cache_key: str, model: str, prompt_version: str, context: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO context_cache VALUES (?, ?, ?, ?, ?)",
                (cache_key, model, prompt_version, context, _now()),
            )

    def activate_version(self, document: Document, version: DocumentVersion, chunks: Sequence[ChildChunk], source_path: str) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO documents(document_id,logical_source,title,mime_type,active,current_version_id,created_at,updated_at,organization_id,created_by,uploaded_by,trust_state)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET title=excluded.title,mime_type=excluded.mime_type,active=1,current_version_id=excluded.current_version_id,updated_at=excluded.updated_at,trust_state=excluded.trust_state""",
                (document.document_id,
                 document.logical_source if document.organization_id == "org_public" else f"{document.organization_id}::{document.logical_source}",
                 document.title, document.mime_type, version.document_version_id, now, now,
                 document.organization_id, document.created_by, document.uploaded_by, document.trust_state),
            )
            connection.execute("UPDATE document_versions SET active=0 WHERE document_id=? AND organization_id=?", (document.document_id, self.organization_id))
            connection.execute(
                """INSERT INTO document_versions(document_version_id,document_id,source_sha256,source_version,size_bytes,source_path,active,created_at,organization_id)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?) ON CONFLICT(document_version_id) DO UPDATE SET active=1,source_path=excluded.source_path""",
                (version.document_version_id, document.document_id, version.source_sha256, version.source_version, version.size_bytes, source_path, now, self.organization_id),
            )
            connection.execute("UPDATE chunks SET active=0 WHERE document_id=? AND organization_id=?", (document.document_id, self.organization_id))
            for chunk in chunks:
                connection.execute(
                    """INSERT INTO chunks(chunk_id,document_id,document_version_id,content_fingerprint,payload_json,active,organization_id)
                    VALUES (?, ?, ?, ?, ?, 1, ?) ON CONFLICT(chunk_id) DO UPDATE SET document_version_id=excluded.document_version_id,content_fingerprint=excluded.content_fingerprint,payload_json=excluded.payload_json,active=1""",
                    (chunk.chunk_id, chunk.document_id, chunk.document_version_id, chunk.content_fingerprint, json.dumps(chunk.to_dict(), ensure_ascii=False), self.organization_id),
                )

    def document_state(self, document_id: str) -> Dict[str, Any] | None:
        with self.connect() as connection:
            document = connection.execute("SELECT active,current_version_id FROM documents WHERE document_id=? AND organization_id=?", (document_id, self.organization_id)).fetchone()
            if not document:
                return None
            chunks = connection.execute("SELECT chunk_id FROM chunks WHERE document_id=? AND active=1 ORDER BY chunk_id", (document_id,)).fetchall()
        return {"active": bool(document["active"]), "current_version_id": document["current_version_id"], "chunk_ids": [row[0] for row in chunks]}

    def restore_document_state(self, document_id: str, state: Dict[str, Any] | None) -> None:
        with self.transaction() as connection:
            connection.execute("UPDATE chunks SET active=0 WHERE document_id=?", (document_id,))
            connection.execute("UPDATE document_versions SET active=0 WHERE document_id=?", (document_id,))
            if state is None:
                connection.execute("UPDATE documents SET active=0,current_version_id=NULL,updated_at=? WHERE document_id=?", (_now(), document_id))
                return
            connection.execute(
                "UPDATE documents SET active=?,current_version_id=?,updated_at=? WHERE document_id=?",
                (int(state["active"]), state["current_version_id"], _now(), document_id),
            )
            if state["current_version_id"]:
                connection.execute("UPDATE document_versions SET active=1 WHERE document_version_id=?", (state["current_version_id"],))
            for chunk_id in state["chunk_ids"]:
                connection.execute("UPDATE chunks SET active=1 WHERE chunk_id=?", (chunk_id,))

    def active_items_and_vectors(self, model: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT c.payload_json, e.dimension, e.vector FROM chunks c JOIN embedding_cache e ON e.content_fingerprint=c.content_fingerprint AND e.embedding_model=? JOIN documents d ON d.document_id=c.document_id WHERE c.active=1 AND d.active=1 AND c.organization_id=? AND d.organization_id=c.organization_id ORDER BY c.chunk_id",
                (model, self.organization_id),
            ).fetchall()
            expected = connection.execute(
                "SELECT COUNT(*) FROM chunks c JOIN documents d ON d.document_id=c.document_id WHERE c.active=1 AND d.active=1 AND c.organization_id=? AND d.organization_id=c.organization_id", (self.organization_id,)
            ).fetchone()[0]
        if len(rows) != expected:
            raise RuntimeError(f"Active chunk/vector mismatch: {expected} chunks and {len(rows)} vectors.")
        items = [json.loads(row["payload_json"]) for row in rows]
        if not rows:
            return items, np.empty((0, 0), dtype="float32")
        dimensions = {row["dimension"] for row in rows}
        if len(dimensions) != 1:
            raise RuntimeError("Cached embeddings have inconsistent dimensions.")
        vectors = np.vstack([np.frombuffer(row["vector"], dtype="float32") for row in rows]).astype("float32")
        return items, vectors

    def active_chunk_payloads(self) -> List[Dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT c.payload_json FROM chunks c JOIN documents d ON d.document_id=c.document_id WHERE c.active=1 AND d.active=1 AND c.organization_id=? AND d.organization_id=c.organization_id ORDER BY c.chunk_id",
                (self.organization_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def list_documents(self) -> List[Dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("""
                SELECT d.*, v.source_sha256, v.source_version, v.size_bytes,
                       (SELECT COUNT(*) FROM chunks c WHERE c.document_id=d.document_id AND c.active=1) AS chunk_count
                FROM documents d LEFT JOIN document_versions v ON v.document_version_id=d.current_version_id
                WHERE d.organization_id=? ORDER BY d.logical_source
            """, (self.organization_id,)).fetchall()
        return [dict(row) for row in rows]

    def document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM chunks WHERE document_id=? AND organization_id=? AND active=1 ORDER BY chunk_id", (document_id, self.organization_id)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def soft_delete_document(self, document_id: str) -> bool:
        with self.transaction() as connection:
            exists = connection.execute("SELECT 1 FROM documents WHERE document_id=? AND organization_id=? AND active=1", (document_id, self.organization_id)).fetchone()
            if not exists:
                return False
            connection.execute("UPDATE documents SET active=0, updated_at=? WHERE document_id=? AND organization_id=?", (_now(), document_id, self.organization_id))
            connection.execute("UPDATE document_versions SET active=0 WHERE document_id=? AND organization_id=?", (document_id, self.organization_id))
            connection.execute("UPDATE chunks SET active=0 WHERE document_id=? AND organization_id=?", (document_id, self.organization_id))
        return True

    def enqueue_job(self, payload: Dict[str, Any], created_by: str = "") -> str:
        job_id, now = "job_" + uuid.uuid4().hex, _now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO ingestion_jobs(job_id,status,stage,progress,payload_json,created_at,updated_at,organization_id,created_by) VALUES (?, 'queued', 'queued', 0, ?, ?, ?, ?, ?)",
                (job_id, json.dumps(payload, ensure_ascii=False), now, now, self.organization_id, created_by),
            )
            connection.execute("INSERT INTO job_events(job_id,stage,progress,message,created_at) VALUES (?, 'queued', 0, 'Job queued', ?)", (job_id, now))
        return job_id

    def recover_stale_jobs(self) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE ingestion_jobs SET status='queued', stage='retrying', lease_until=NULL, updated_at=? WHERE status='running' AND lease_until<?",
                (_now(), _now()),
            )
        return cursor.rowcount

    def claim_job(self, lease_seconds: int) -> Dict[str, Any] | None:
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM ingestion_jobs WHERE status='queued' AND organization_id=? ORDER BY created_at LIMIT 1", (self.organization_id,)).fetchone()
            if not row:
                return None
            now = _now()
            connection.execute(
                "UPDATE ingestion_jobs SET status='running', stage='validating', progress=0.02, attempts=attempts+1, lease_until=?, updated_at=? WHERE job_id=?",
                (now + lease_seconds, now, row["job_id"]),
            )
            result = dict(row)
            result.update({"status": "running", "stage": "validating", "progress": 0.02, "payload": json.loads(row["payload_json"])})
            return result

    def update_job(self, job_id: str, stage: str, progress: float, message: str, lease_seconds: int = 120) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE ingestion_jobs SET stage=?, progress=?, lease_until=?, updated_at=? WHERE job_id=?",
                (stage, progress, now + lease_seconds, now, job_id),
            )
            connection.execute("INSERT INTO job_events(job_id,stage,progress,message,created_at) VALUES (?, ?, ?, ?, ?)", (job_id, stage, progress, message, now))

    def finish_job(self, job_id: str, result: Dict[str, Any], warnings: Sequence[Dict[str, str]]) -> None:
        status = "succeeded_with_warnings" if warnings else "succeeded"
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE ingestion_jobs SET status=?, stage='complete', progress=1, result_json=?, warnings_json=?, lease_until=NULL, updated_at=? WHERE job_id=?",
                (status, json.dumps(result, ensure_ascii=False), json.dumps(list(warnings), ensure_ascii=False), now, job_id),
            )
            connection.execute("INSERT INTO job_events(job_id,stage,progress,message,created_at) VALUES (?, 'complete', 1, ?, ?)", (job_id, status, now))

    def fail_job(self, job_id: str, code: str, message: str) -> None:
        now = _now()
        with self.transaction() as connection:
            connection.execute(
                "UPDATE ingestion_jobs SET status='failed', stage='failed', error_code=?, error_message=?, lease_until=NULL, updated_at=? WHERE job_id=?",
                (code, message, now, job_id),
            )
            connection.execute("INSERT INTO job_events(job_id,stage,progress,message,created_at) VALUES (?, 'failed', 1, ?, ?)", (job_id, message, now))

    def get_job(self, job_id: str) -> Dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM ingestion_jobs WHERE job_id=? AND organization_id=?", (job_id, self.organization_id)).fetchone()
            events = connection.execute("SELECT stage,progress,message,created_at FROM job_events WHERE job_id=? ORDER BY event_id", (job_id,)).fetchall() if row else []
        if not row:
            return None
        result = dict(row)
        for key in ("payload_json", "result_json", "warnings_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key) or ("[]" if key == "warnings_json" else "null"))
        result["events"] = [dict(event) for event in events]
        return result

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT job_id,status,stage,progress,result_json,warnings_json,error_code,error_message,attempts,created_at,updated_at,created_by FROM ingestion_jobs WHERE organization_id=? ORDER BY created_at DESC LIMIT ?",
                (self.organization_id, limit),
            ).fetchall()
        jobs = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item.pop("result_json") or "null")
            item["warnings"] = json.loads(item.pop("warnings_json") or "[]")
            jobs.append(item)
        return jobs

    def retry_job(self, job_id: str, max_retries: int) -> bool:
        with self.transaction() as connection:
            row = connection.execute("SELECT attempts,status FROM ingestion_jobs WHERE job_id=? AND organization_id=?", (job_id, self.organization_id)).fetchone()
            if not row or row["status"] != "failed" or row["attempts"] >= max_retries:
                return False
            connection.execute(
                "UPDATE ingestion_jobs SET status='queued',stage='queued',progress=0,error_code=NULL,error_message=NULL,updated_at=? WHERE job_id=? AND organization_id=?",
                (_now(), job_id, self.organization_id),
            )
        return True

    def cancel_job(self, job_id: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE ingestion_jobs SET status='cancelled',stage='cancelled',progress=1,updated_at=? WHERE job_id=? AND organization_id=? AND status='queued'",
                (_now(), job_id, self.organization_id),
            )
            if cursor.rowcount:
                connection.execute(
                    "INSERT INTO job_events(job_id,stage,progress,message,created_at) VALUES (?, 'cancelled', 1, 'Job cancelled', ?)",
                    (job_id, _now()),
                )
        return bool(cursor.rowcount)
