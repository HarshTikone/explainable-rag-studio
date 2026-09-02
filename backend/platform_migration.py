"""Idempotent SQLite/FAISS to PostgreSQL/pgvector cutover tooling."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select, text

from .database import DatabaseRuntime, Organization
from .postgres_store import PgVectorStore
from .security_models import ROLE_SCOPES, SecurityBoundaryError, SecurityContext
from .vectorstore import FaissStore


@dataclass(frozen=True)
class TenantMigrationPlan:
    organization_id: str
    source_index: str
    chunks: int
    vector_dimension: int
    corpus_fingerprint: str


class PlatformMigrator:
    def __init__(self, database: DatabaseRuntime, repository_root: str):
        self.database = database
        self.root = Path(repository_root).resolve()

    def plan(self, mapping: dict[str, str] | None = None) -> dict[str, Any]:
        mapping = mapping or {}
        candidates: list[tuple[str, Path]] = []
        legacy = self.root / "index"
        if (legacy / "faiss.index").exists():
            candidates.append(("org_public", legacy))
        tenant_root = legacy / "organizations"
        if tenant_root.exists():
            for path in sorted(item for item in tenant_root.iterdir() if item.is_dir()):
                candidates.append((mapping.get(path.name, path.name), path))
        plans = []
        for organization_id, path in candidates:
            store = FaissStore(str(path))
            if not store.load():
                continue
            items = store.meta.get("items", [])
            for item in items:
                item_org = item.get("organization_id")
                if item_org is None and organization_id == "org_public":
                    continue
                if item_org != organization_id:
                    raise SecurityBoundaryError(f"Mixed-tenant source index: {path}")
            canonical = json.dumps(items, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            plans.append(TenantMigrationPlan(
                organization_id, str(path), len(items), int(store.index.d), hashlib.sha256(canonical.encode()).hexdigest()
            ))
        payload = [asdict(item) for item in plans]
        return {"schema_version": "1.0", "mode": "dry_run", "tenants": payload,
                "fingerprint": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}

    def execute(self, plan: dict[str, Any]) -> dict[str, Any]:
        migrated = []
        for tenant in plan["tenants"]:
            organization_id = tenant["organization_id"]
            with self.database.engine.begin() as connection:
                connection.execute(text(
                    "INSERT INTO organizations (organization_id,name,active,created_at) "
                    "VALUES (:id,:name,true,now()) ON CONFLICT (organization_id) DO NOTHING"
                ), {"id": organization_id, "name": "Public Demo" if organization_id == "org_public" else organization_id})
            context = SecurityContext(organization_id, "migration", "owner", None, ROLE_SCOPES["owner"])
            source = FaissStore(tenant["source_index"])
            if not source.load():
                raise RuntimeError(f"Source index disappeared: {tenant['source_index']}")
            items = []
            for original in source.meta.get("items", []):
                item = dict(original)
                item.setdefault("organization_id", organization_id)
                item.setdefault("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
                items.append(item)
            vectors = np.empty((source.index.ntotal, source.index.d), dtype="float32")
            if source.index.ntotal:
                source.index.reconstruct_n(0, source.index.ntotal, vectors)
            target = PgVectorStore(self.database, context)
            target.replace_active_chunks(items, vectors)
            if len(target.meta["items"]) != len(items):
                raise RuntimeError(f"PostgreSQL parity check failed for {organization_id}.")
            migrated.append({"organization_id": organization_id, "chunks": len(items), "validated": True})
        return {**plan, "mode": "executed", "migrated": migrated, "rollback": "legacy_read_only"}

    def validate(self, plan: dict[str, Any]) -> dict[str, Any]:
        results = []
        for tenant in plan["tenants"]:
            context = SecurityContext(tenant["organization_id"], "migration", "owner", None, ROLE_SCOPES["owner"])
            target = PgVectorStore(self.database, context)
            actual = len(target.meta["items"])
            results.append({"organization_id": tenant["organization_id"], "expected": tenant["chunks"],
                            "actual": actual, "matched": actual == tenant["chunks"]})
        return {"valid": all(item["matched"] for item in results), "tenants": results,
                "source_plan_fingerprint": plan["fingerprint"]}
