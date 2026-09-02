"""Migrate a provably sanitized legacy index into the reserved public tenant."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import SETTINGS
from backend.ingestion import MANIFEST_FILE, MANIFEST_SCHEMA_VERSION
from backend.vectorstore import FaissStore


def migrate_public(source: Path, target: Path, mapping_file: Path | None = None) -> dict:
    legacy = FaissStore(str(source))
    if not legacy.load():
        raise RuntimeError("Legacy index is not readable.")
    items = legacy.meta.get("items", [])
    mapping = json.loads(mapping_file.read_text("utf-8")) if mapping_file else {}
    private = [item for item in items if item.get("organization_id") not in (None, SETTINGS.public_organization_id)]
    if private and not mapping:
        raise RuntimeError("Legacy records cannot be proven public; provide an explicit organization mapping file.")
    if any(mapping.get(item.get("document_id") or item.get("source"), SETTINGS.public_organization_id) != SETTINGS.public_organization_id for item in items):
        raise RuntimeError("Private mappings require a separate per-organization rebuild; no mixed index will be created.")
    migrated = [{**item, "organization_id": SETTINGS.public_organization_id, "trust_state": "public"} for item in items]
    vectors = np.vstack([legacy.index.reconstruct(index) for index in range(legacy.index.ntotal)]).astype("float32")
    if target.exists():
        existing = FaissStore(str(target))
        if existing.load() and existing.meta.get("items") == migrated:
            return {"status": "unchanged", "chunks": len(migrated), "target": str(target)}
        raise RuntimeError("Target tenant already exists with different content.")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / ("." + target.name + "-migration")
    if staging.exists():
        shutil.rmtree(staging)
    built = FaissStore(str(staging))
    built.build(vectors, migrated)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION, "organization_id": SETTINGS.public_organization_id,
        "isolation_capability": "per_organization_faiss", "security_policy_version": SETTINGS.security_policy_version,
        "legacy_read_only_migration": True, "authorized_document_versions": sorted({item.get("document_version_id") for item in migrated if item.get("document_version_id")}),
    }
    (staging / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    staging.replace(target)
    return {"status": "migrated", "chunks": len(migrated), "target": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SETTINGS.index_dir)
    parser.add_argument("--mapping")
    args = parser.parse_args()
    target = Path(SETTINGS.tenant_index_root) / SETTINGS.public_organization_id
    print(json.dumps(migrate_public(Path(args.source), target, Path(args.mapping) if args.mapping else None), indent=2))


if __name__ == "__main__":
    main()
