"""PostgreSQL persistence and row-level-security transaction boundaries.

SQLite remains available only to the explicit legacy/read-only migration path.
Every protected PostgreSQL transaction receives organization and actor context
before any tenant-owned table can be read.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .security_models import SecurityContext


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Organization(Base):
    __tablename__ = "organizations"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    oidc_subject: Mapped[str | None] = mapped_column(String(255), unique=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    display_name: Mapped[str] = mapped_column(String(160))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Membership(Base):
    __tablename__ = "memberships"
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.organization_id"), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"
    key_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.organization_id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"))
    digest: Mapped[str] = mapped_column(String(64))
    scopes: Mapped[list[str]] = mapped_column(JSONB)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GlobalApiKeyLookup(Base):
    """Non-secret routing record needed to establish tenant context before RLS."""
    __tablename__ = "global_api_key_lookup"
    key_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), index=True)


class GlobalIdentityLookup(Base):
    """OIDC subject-to-membership routing; authorization still occurs under RLS."""
    __tablename__ = "global_identity_lookup"
    issuer_subject_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64))


class Document(Base):
    __tablename__ = "documents"
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.organization_id"), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    logical_source: Mapped[str] = mapped_column(String(512))
    title: Mapped[str] = mapped_column(String(512))
    uploaded_by: Mapped[str] = mapped_column(String(64))
    trust_state: Mapped[str] = mapped_column(String(24), default="untrusted")
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_version_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(96), index=True)
    source_sha256: Mapped[str] = mapped_column(String(64))
    source_version: Mapped[str] = mapped_column(String(128), default="")
    object_key: Mapped[str | None] = mapped_column(String(1024))
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        UniqueConstraint("organization_id", "document_id", "source_sha256", name="uq_document_source_version"),
    )


class Chunk(Base):
    __tablename__ = "chunks"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(96), index=True)
    document_version_id: Mapped[str] = mapped_column(String(96), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text)
    retrieval_text: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    content_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    embedding_model: Mapped[str] = mapped_column(String(255))
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    created_by: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    progress_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GlobalJobLookup(Base):
    """Non-secret worker routing record used before tenant RLS is established."""
    __tablename__ = "global_job_lookup"
    job_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), index=True)
    created_by: Mapped[str] = mapped_column(String(64))


class JobEvent(Base):
    __tablename__ = "job_events"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(96), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    progress: Mapped[int] = mapped_column(Integer)
    message: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReviewCase(Base):
    __tablename__ = "review_cases"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), index=True)
    fingerprint: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("organization_id", "fingerprint", name="uq_review_case_fingerprint"),)


class ReviewerDecision(Base):
    __tablename__ = "reviewer_decisions"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    decision_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(96), index=True)
    reviewer_id: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(24))
    notes: Mapped[str] = mapped_column(String(1000), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class QuarantineCase(Base):
    __tablename__ = "quarantine_cases"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    job_id: Mapped[str | None] = mapped_column(String(96))
    document_name: Mapped[str] = mapped_column(String(512))
    object_key: Mapped[str] = mapped_column(String(1024))
    status: Mapped[str] = mapped_column(String(24), default="open", index=True)
    findings_json: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    created_by: Mapped[str] = mapped_column(String(64))
    resolved_by: Mapped[str | None] = mapped_column(String(64))
    resolution_reason: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ObjectRecord(Base):
    __tablename__ = "object_records"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    object_key: Mapped[str] = mapped_column(String(1024), primary_key=True)
    version_id: Mapped[str | None] = mapped_column(String(255))
    checksum_sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    content_type: Mapped[str] = mapped_column(String(255))
    encryption_algorithm: Mapped[str] = mapped_column(String(64))
    encrypted_data_key: Mapped[bytes | None] = mapped_column(LargeBinary)
    nonce: Mapped[bytes | None] = mapped_column(LargeBinary)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    purge_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class QuotaUsage(Base):
    __tablename__ = "quota_usage"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    period: Mapped[str] = mapped_column(String(16), primary_key=True)
    query_count: Mapped[int] = mapped_column(BigInteger, default=0)
    upload_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    embedding_units: Mapped[int] = mapped_column(BigInteger, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ExperimentRecord(Base):
    __tablename__ = "experiment_records"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    created_by: Mapped[str] = mapped_column(String(64))
    manifest_json: Mapped[dict[str, Any]] = mapped_column(JSONB)
    artifact_key: Mapped[str | None] = mapped_column(String(1024))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TelemetryRecord(Base):
    __tablename__ = "telemetry_records"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    telemetry_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(96), index=True)
    query_sha256: Mapped[str] = mapped_column(String(64))
    query_length: Mapped[int] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(String(64), default="query")
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RetentionRecord(Base):
    __tablename__ = "retention_records"
    organization_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    record_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    object_type: Mapped[str] = mapped_column(String(64))
    object_id: Mapped[str] = mapped_column(String(128))
    retain_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    legal_hold: Mapped[bool] = mapped_column(Boolean, default=False)
    purged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    __tablename__ = "audit_events"
    sequence: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(96), unique=True)
    organization_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str | None] = mapped_column(String(64))
    key_id: Mapped[str | None] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(128))
    object_type: Mapped[str] = mapped_column(String(64))
    object_id: Mapped[str | None] = mapped_column(String(128))
    result: Mapped[str] = mapped_column(String(32))
    reason_code: Mapped[str] = mapped_column(String(64), default="")
    http_status: Mapped[int] = mapped_column(Integer)
    request_id: Mapped[str] = mapped_column(String(96), default="")
    details_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    previous_hash: Mapped[str] = mapped_column(String(64))
    event_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


TENANT_TABLES = (
    "memberships", "api_keys", "documents", "document_versions", "chunks",
    "ingestion_jobs", "job_events", "review_cases", "reviewer_decisions",
    "quarantine_cases", "object_records", "quota_usage", "experiment_records",
    "telemetry_records", "retention_records", "audit_events",
)


def rls_statements() -> list[str]:
    """Return deterministic, idempotent deny-by-default RLS DDL."""
    statements: list[str] = []
    for table in TENANT_TABLES:
        policy = f"{table}_tenant_isolation"
        statements.extend([
            f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY',
            f'ALTER TABLE "{table}" FORCE ROW LEVEL SECURITY',
            f'DROP POLICY IF EXISTS "{policy}" ON "{table}"',
            (
                f'CREATE POLICY "{policy}" ON "{table}" '
                "USING (organization_id = NULLIF(current_setting('app.organization_id', true), '')) "
                "WITH CHECK (organization_id = NULLIF(current_setting('app.organization_id', true), ''))"
            ),
        ])
    return statements


class DatabaseRuntime:
    def __init__(self, database_url: str, *, echo: bool = False):
        if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("Production DATABASE_URL must use PostgreSQL.")
        self.engine = create_engine(database_url, pool_pre_ping=True, echo=echo)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def ping(self) -> bool:
        with self.engine.connect() as connection:
            return connection.execute(text("SELECT 1")).scalar_one() == 1

    @contextmanager
    def session(self, context: SecurityContext, *, write: bool = False) -> Iterator[Session]:
        session = self.sessions()
        try:
            session.execute(text("SELECT set_config('app.organization_id', :value, true)"), {"value": context.organization_id})
            session.execute(text("SELECT set_config('app.actor_user_id', :value, true)"), {"value": context.user_id})
            session.execute(text("SET LOCAL statement_timeout = '30s'"))
            yield session
            if write:
                session.commit()
            else:
                session.rollback()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def install_schema(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            Base.metadata.create_all(connection)
            for statement in rls_statements():
                connection.execute(text(statement))

    def migration_fingerprint(self) -> str:
        metadata = sorted((table.name, sorted(column.name for column in table.columns)) for table in Base.metadata.tables.values())
        return hashlib.sha256(json.dumps(metadata, separators=(",", ":")).encode()).hexdigest()


@event.listens_for(AuditEvent, "before_update")
def _reject_audit_update(*_args) -> None:
    raise ValueError("Audit events are append-only.")


@event.listens_for(AuditEvent, "before_delete")
def _reject_audit_delete(*_args) -> None:
    raise ValueError("Audit events are append-only.")
