"""pgvector adapter with transaction-scoped tenant enforcement."""
from __future__ import annotations

from threading import RLock
from typing import Any, Iterable

import numpy as np
from sqlalchemy import delete, select, update

from .database import Chunk, DatabaseRuntime, Document, DocumentVersion
from .security_models import RetrievalScope, SecurityBoundaryError, SecurityContext


class PgVectorStore:
    """FAISS-compatible read surface backed by PostgreSQL and pgvector."""

    def __init__(self, database: DatabaseRuntime, context: SecurityContext):
        if context.organization_id == "":
            raise ValueError("A tenant context is required.")
        self.database = database
        self.context = context
        self.index = self  # compatibility readiness marker
        self._meta: dict[str, list[dict[str, Any]]] | None = None
        self._lock = RLock()

    @property
    def meta(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            if self._meta is None:
                with self.database.session(self.context) as session:
                    rows = session.execute(
                        select(Chunk).where(Chunk.active.is_(True)).order_by(Chunk.chunk_id)
                    ).scalars().all()
                    self._meta = {"items": [self._item(row) for row in rows]}
            return self._meta

    def invalidate(self) -> None:
        with self._lock:
            self._meta = None

    def load(self) -> bool:
        self.invalidate()
        return True

    def search(self, query_vec: np.ndarray, top_k: int):
        if top_k <= 0:
            return []
        vector = np.asarray(query_vec, dtype="float32").reshape(-1).tolist()
        distance = Chunk.embedding.cosine_distance(vector)
        with self.database.session(self.context) as session:
            rows = session.execute(
                select(Chunk, distance.label("distance"))
                .where(Chunk.active.is_(True))
                .order_by(distance, Chunk.chunk_id)
                .limit(top_k)
            ).all()
        result = []
        for row, raw_distance in rows:
            item = self._item(row)
            if item["organization_id"] != self.context.organization_id:
                raise SecurityBoundaryError("PostgreSQL returned cross-tenant vector metadata.")
            result.append((1.0 - float(raw_distance), item))
        return result

    def replace_active_chunks(self, items: Iterable[dict[str, Any]], vectors: np.ndarray) -> None:
        records = list(items)
        matrix = np.asarray(vectors, dtype="float32")
        if len(records) != len(matrix):
            raise ValueError("Vector/metadata alignment mismatch.")
        if any(item.get("organization_id") != self.context.organization_id for item in records):
            raise SecurityBoundaryError("Cannot write chunks for another organization.")
        with self.database.session(self.context, write=True) as session:
            session.execute(update(Chunk).where(Chunk.active.is_(True)).values(active=False))
            for item, vector in zip(records, matrix):
                existing = session.get(Chunk, (self.context.organization_id, item["chunk_id"]))
                values = {
                    "document_id": item.get("document_id", ""),
                    "document_version_id": item.get("document_version_id", ""),
                    "parent_id": item.get("parent_id"),
                    "text": item.get("text", ""),
                    "retrieval_text": item.get("retrieval_text", item.get("text", "")),
                    "metadata_json": {key: value for key, value in item.items() if key not in {"text", "retrieval_text"}},
                    "content_fingerprint": item.get("content_fingerprint", ""),
                    "embedding_model": item.get("embedding_model", ""),
                    "embedding": vector.tolist(),
                    "active": True,
                }
                if existing:
                    for key, value in values.items():
                        setattr(existing, key, value)
                else:
                    session.add(Chunk(
                        organization_id=self.context.organization_id,
                        chunk_id=item["chunk_id"],
                        **values,
                    ))
        self.invalidate()

    def upsert_document_chunks(self, document, version, items: Iterable[dict[str, Any]], vectors: np.ndarray, object_key: str | None = None) -> str:
        """Idempotently activate one document version without touching unrelated chunks."""
        records = list(items)
        matrix = np.asarray(vectors, dtype="float32")
        if len(records) != len(matrix):
            raise ValueError("Vector/metadata alignment mismatch.")
        if any(item.get("organization_id") != self.context.organization_id for item in records):
            raise SecurityBoundaryError("Cannot write chunks for another organization.")
        with self.database.session(self.context, write=True) as session:
            current = session.get(DocumentVersion, (self.context.organization_id, version.document_version_id))
            if current and current.active:
                return "unchanged"
            document_row = session.get(Document, (self.context.organization_id, document.document_id))
            was_existing = document_row is not None
            if document_row is None:
                document_row = Document(
                    organization_id=self.context.organization_id,
                    document_id=document.document_id,
                    logical_source=document.logical_source,
                    title=document.title,
                    uploaded_by=document.uploaded_by or self.context.user_id,
                    trust_state=document.trust_state,
                )
                session.add(document_row)
            else:
                document_row.title = document.title
                document_row.trust_state = document.trust_state
                document_row.active = True
                document_row.deleted_at = None
            session.execute(
                update(DocumentVersion)
                .where(DocumentVersion.document_id == document.document_id, DocumentVersion.active.is_(True))
                .values(active=False)
            )
            session.execute(
                update(Chunk)
                .where(Chunk.document_id == document.document_id, Chunk.active.is_(True))
                .values(active=False)
            )
            if current is None:
                session.add(DocumentVersion(
                    organization_id=self.context.organization_id,
                    document_version_id=version.document_version_id,
                    document_id=document.document_id,
                    source_sha256=version.source_sha256,
                    source_version=version.source_version,
                    object_key=object_key,
                    active=True,
                ))
            else:
                current.active = True
                current.object_key = object_key or current.object_key
            for item, vector in zip(records, matrix):
                existing = session.get(Chunk, (self.context.organization_id, item["chunk_id"]))
                values = {
                    "document_id": item.get("document_id", ""),
                    "document_version_id": item.get("document_version_id", ""),
                    "parent_id": item.get("parent_id"),
                    "text": item.get("text", ""),
                    "retrieval_text": item.get("retrieval_text", item.get("text", "")),
                    "metadata_json": {key: value for key, value in item.items() if key not in {"text", "retrieval_text"}},
                    "content_fingerprint": item.get("content_fingerprint", ""),
                    "embedding_model": item.get("embedding_model", ""),
                    "embedding": vector.tolist(),
                    "active": True,
                }
                if existing:
                    for key, value in values.items():
                        setattr(existing, key, value)
                else:
                    session.add(Chunk(organization_id=self.context.organization_id, chunk_id=item["chunk_id"], **values))
        self.invalidate()
        return "updated" if was_existing else "indexed"

    @staticmethod
    def _item(row: Chunk) -> dict[str, Any]:
        return {
            **dict(row.metadata_json or {}),
            "organization_id": row.organization_id,
            "chunk_id": row.chunk_id,
            "document_id": row.document_id,
            "document_version_id": row.document_version_id,
            "parent_id": row.parent_id,
            "text": row.text,
            "retrieval_text": row.retrieval_text,
            "content_fingerprint": row.content_fingerprint,
            "embedding_model": row.embedding_model,
        }
