import numpy as np
import pytest

from backend.retriever import retrieve
from backend.security_models import RetrievalScope, SecurityBoundaryError
from backend.tenant_store import TenantStoreManager
from backend.document_parsers import stable_document_id
from scripts.migrate_tenants import migrate_public


class Embedder:
    def embed_query(self, _query):
        return np.asarray([[1.0, 0.0]], dtype="float32")

    def embed_texts(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")


class Reranker:
    def score(self, query, documents):
        return [1.0 for _ in documents]


def item(chunk_id, organization_id, text="shared ID TS-999"):
    return {"chunk_id": chunk_id, "organization_id": organization_id, "text": text, "retrieval_text": text,
            "generation_text": text, "source": "same.md"}


def test_physical_tenant_stores_and_all_strategies(tmp_path):
    manager = TenantStoreManager(str(tmp_path / "organizations"), 2)
    first = manager.get("org_a")
    first.build(np.asarray([[1.0, 0.0]], dtype="float32"), [item("a1", "org_a")])
    second = manager.get("org_b")
    second.build(np.asarray([[1.0, 0.0]], dtype="float32"), [item("b1", "org_b")])
    for strategy in ("dense", "dense_mmr", "hybrid_rrf", "hybrid_rerank"):
        result = retrieve(manager.get("org_a", reload=True), Embedder(), "TS-999", 1, strategy,
                          reranker=Reranker(), scope=RetrievalScope("org_a", "user_a"))
        assert [hit.item["chunk_id"] for hit in result.hits] == ["a1"]
        assert result.organization_id == "org_a"


def test_mixed_metadata_fails_before_dense_or_lexical_scoring(tmp_path):
    manager = TenantStoreManager(str(tmp_path / "organizations"))
    mixed = manager.get("org_a")
    mixed.build(np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype="float32"), [item("a", "org_a"), item("canary", "org_b")])
    with pytest.raises(SecurityBoundaryError):
        retrieve(mixed, Embedder(), "canary", 2, "hybrid_rrf", scope=RetrievalScope("org_a", "user_a"))


def test_store_manager_rejects_path_escape(tmp_path):
    manager = TenantStoreManager(str(tmp_path / "organizations"))
    with pytest.raises(ValueError):
        manager.get("../other")


def test_store_cache_eviction_and_invalidation(tmp_path):
    manager = TenantStoreManager(str(tmp_path / "organizations"), 1)
    manager.get("org_a")
    manager.get("org_b")
    assert list(manager._stores) == ["org_b"]
    manager.invalidate("org_b")
    assert not manager._stores


def test_private_document_ids_are_namespaced_and_public_migration_is_idempotent(tmp_path):
    assert stable_document_id("same.md", "org_a") != stable_document_id("same.md", "org_b")
    legacy = TenantStoreManager(str(tmp_path / "legacy-root")).get("legacy")
    legacy.build(np.asarray([[1.0, 0.0]], dtype="float32"), [{"chunk_id": "c000001", "text": "public", "source": "public.md"}])
    target = tmp_path / "organizations" / "org_public"
    first = migrate_public(tmp_path / "legacy-root" / "legacy", target)
    second = migrate_public(tmp_path / "legacy-root" / "legacy", target)
    assert first["status"] == "migrated" and second["status"] == "unchanged"
    migrated = TenantStoreManager(str(tmp_path / "organizations")).get("org_public")
    assert migrated.meta["items"][0]["chunk_id"] == "c000001"
    assert migrated.meta["items"][0]["organization_id"] == "org_public"
