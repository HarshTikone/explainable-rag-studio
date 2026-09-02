"""Encrypted backup, restore verification, and balanced retention operations."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select

from .database import AuditEvent, Chunk, DatabaseRuntime, ObjectRecord, Organization, RetentionRecord


def create_encrypted_backup(database_url: str, object_store, organization_id: str = "org_platform") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rag-backup-") as directory:
        dump = Path(directory) / "database.dump"
        environment = dict(os.environ)
        completed = subprocess.run(["pg_dump", "--format=custom", "--file", str(dump), database_url],
                                   env=environment, capture_output=True, text=True, timeout=3600)
        if completed.returncode:
            raise RuntimeError("pg_dump failed: " + completed.stderr[-1000:])
        payload = dump.read_bytes()
        stored = object_store.put(organization_id, "database.dump", payload, "application/octet-stream",
                                  {"kind": "postgres_backup", "created_at": datetime.now(timezone.utc).isoformat()})
        return {"schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(),
                "object_key": stored.object_key, "version_id": stored.version_id,
                "checksum_sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
                "rpo_hours": 24, "encrypted": True}


def restore_and_validate(source_url: str, restore_url: str, dump_bytes: bytes) -> dict[str, Any]:
    if source_url == restore_url:
        raise ValueError("Restore validation must target an isolated database.")
    with tempfile.TemporaryDirectory(prefix="rag-restore-") as directory:
        dump = Path(directory) / "database.dump"
        dump.write_bytes(dump_bytes)
        started = datetime.now(timezone.utc)
        completed = subprocess.run(["pg_restore", "--clean", "--if-exists", "--no-owner", "--dbname", restore_url, str(dump)],
                                   capture_output=True, text=True, timeout=4 * 3600)
        if completed.returncode:
            raise RuntimeError("pg_restore failed: " + completed.stderr[-1000:])
        restored = DatabaseRuntime(restore_url)
        with restored.engine.connect() as connection:
            organizations = connection.execute(select(func.count()).select_from(Organization)).scalar_one()
            chunks = connection.execute(select(func.count()).select_from(Chunk)).scalar_one()
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        return {"valid": restored.ping() and organizations >= 1, "organizations": organizations,
                "chunks": chunks, "restore_seconds": elapsed, "rto_target_seconds": 14400,
                "rto_passed": elapsed < 14400}


def enforce_retention(database: DatabaseRuntime, context, object_store, now: datetime | None = None) -> dict[str, int]:
    current = now or datetime.now(timezone.utc)
    counts = {"objects": 0, "retention_records": 0}
    with database.session(context, write=True) as session:
        objects = session.execute(select(ObjectRecord).where(ObjectRecord.purge_after <= current, ObjectRecord.deleted_at.is_not(None))).scalars().all()
        for item in objects:
            object_store.soft_delete(context.organization_id, item.object_key)
            session.delete(item)
            counts["objects"] += 1
        records = session.execute(select(RetentionRecord).where(RetentionRecord.retain_until <= current,
                                                                  RetentionRecord.legal_hold.is_(False),
                                                                  RetentionRecord.purged_at.is_(None))).scalars().all()
        for record in records:
            record.purged_at = current
            counts["retention_records"] += 1
    return counts
