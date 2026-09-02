import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from backend.contextual_chunking import build_contextual_chunks
from backend.citations import pick_top_citations
from backend.document_parsers import parse_document, stable_document_id
from backend.ingestion import IngestionOptions, IngestionService, IngestionWorker, activate_faiss_atomically
from backend.ingestion_registry import IngestionRegistry
from backend.retriever import retrieve
from backend.vectorstore import FaissStore


class FakeEmbedder:
    calls = []

    def __init__(self, model):
        self.model = model

    def embed_texts(self, texts):
        self.calls.append((self.model, list(texts)))
        values = [[max(1, len(text)), max(1, sum(ord(char) for char in text) % 997)] for text in texts]
        vectors = np.asarray(values, dtype="float32")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def embed_query(self, text):
        return self.embed_texts([text])


class FakeReranker:
    model_name = "fake"

    def score(self, _query, documents):
        return [float(len(document)) for document in documents]


def make_service(tmp_path):
    registry = IngestionRegistry(str(tmp_path / "outputs" / "ingestion.db"))
    service = IngestionService(registry, str(tmp_path / "index"), str(tmp_path / "uploads"), FakeEmbedder)
    return registry, service


def payload(path, **options):
    values = {"embedding_model": "fake", **options}
    defaults = IngestionOptions(**values)
    return {"path": str(path), "source_name": path.name, "options": defaults.__dict__}


def test_contextual_chunks_are_stable_and_keep_parent_metadata(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Setup\nUse AU-4012.\n\n## Recovery\nSynchronize the tenant clock.", encoding="utf-8")
    parsed = parse_document(str(path))
    first = build_contextual_chunks(parsed.document, parsed.version, parsed.blocks, child_tokens=20, overlap_tokens=2)
    second = build_contextual_chunks(parsed.document, parsed.version, parsed.blocks, child_tokens=20, overlap_tokens=2)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.parent_id and chunk.heading_path for chunk in first)
    assert all(chunk.retrieval_text.startswith("Document:") for chunk in first)
    assert all(chunk.contextual_prefix in chunk.retrieval_text for chunk in first)
    split_parents = {chunk.parent_id for chunk in build_contextual_chunks(parsed.document, parsed.version, parsed.blocks, child_tokens=20, overlap_tokens=2, parent_tokens=3)}
    assert len(split_parents) > 1


def test_ingestion_is_idempotent_updates_only_one_document_and_deletes_cleanly(tmp_path):
    registry, service = make_service(tmp_path)
    alpha = tmp_path / "alpha.md"
    beta = tmp_path / "beta.md"
    alpha.write_text("# Alpha\nCurrent alpha procedure.", encoding="utf-8")
    beta.write_text("# Beta\nStable beta procedure.", encoding="utf-8")

    first = service.process_payload(payload(alpha))
    service.process_payload(payload(beta))
    beta_id = next(document["document_id"] for document in registry.list_documents() if document["logical_source"] == "beta.md")
    beta_chunks_before = [chunk["chunk_id"] for chunk in registry.document_chunks(beta_id)]
    assert service.process_payload(payload(alpha))["outcome"] == "unchanged"

    alpha_id = first["document_id"]
    old_alpha_chunks = {chunk["chunk_id"] for chunk in registry.document_chunks(alpha_id)}
    alpha.write_text("# Alpha\nUpdated alpha procedure with AU-9.", encoding="utf-8")
    assert service.process_payload(payload(alpha))["outcome"] == "updated"
    active_ids = {item["chunk_id"] for item in FaissStore(str(tmp_path / "index")).meta["items"]}
    loaded = FaissStore(str(tmp_path / "index"))
    assert loaded.load()
    active_ids = {item["chunk_id"] for item in loaded.meta["items"]}
    assert not old_alpha_chunks & active_ids
    assert [chunk["chunk_id"] for chunk in registry.document_chunks(beta_id)] == beta_chunks_before

    assert service.delete_document(alpha_id, "fake")
    reloaded = FaissStore(str(tmp_path / "index"))
    assert reloaded.load()
    assert all(item["document_id"] != alpha_id for item in reloaded.meta["items"])
    for strategy in ("dense", "dense_mmr", "hybrid_rrf", "hybrid_rerank"):
        result = retrieve(reloaded, FakeEmbedder("fake"), "alpha procedure", 5, strategy, reranker=FakeReranker())
        assert all(hit.item["document_id"] != alpha_id for hit in result.hits)
        assert all(citation["chunk_id"] not in old_alpha_chunks for citation in pick_top_citations(result))
    assert not service.delete_document(alpha_id, "fake")


def test_embedding_cache_is_reused_and_model_change_reembeds_active_chunks(tmp_path):
    FakeEmbedder.calls.clear()
    registry, service = make_service(tmp_path)
    path = tmp_path / "cache.md"
    path.write_text("# Cache\nReusable content.", encoding="utf-8")
    service.process_payload(payload(path))
    assert len(FakeEmbedder.calls) == 1
    path.write_text("# Cache\nReusable content. New detail.", encoding="utf-8")
    service.process_payload(payload(path))
    assert len(FakeEmbedder.calls) == 2
    second = tmp_path / "second.md"
    second.write_text("# Second\nAnother document.", encoding="utf-8")
    service.process_payload(payload(second, embedding_model="fake-v2"))
    assert FakeEmbedder.calls[-1][0] == "fake-v2"
    assert len(registry.active_items_and_vectors("fake-v2")[0]) == 2


def test_identical_document_repairs_missing_manifest_instead_of_false_unchanged(tmp_path):
    registry, service = make_service(tmp_path)
    path = tmp_path / "repair.md"
    path.write_text("# Repair\nRecover an interrupted activation.", encoding="utf-8")
    service.process_payload(payload(path))
    (tmp_path / "index" / "manifest.json").unlink()
    repaired = service.process_payload(payload(path))
    assert repaired["outcome"] == "updated"
    assert (tmp_path / "index" / "manifest.json").exists()


def test_index_wide_configuration_mismatch_is_rejected(tmp_path):
    _registry, service = make_service(tmp_path)
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# First\nOne.", encoding="utf-8")
    second.write_text("# Second\nTwo.", encoding="utf-8")
    service.process_payload(payload(first))
    with pytest.raises(Exception) as error:
        service.process_payload(payload(second, child_tokens=500))
    assert error.value.code == "INDEX_CONFIG_MISMATCH"


def test_atomic_activation_preserves_previous_index_on_build_failure(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    items = [{"chunk_id": "old", "text": "old"}]
    activate_faiss_atomically(str(index_dir), np.asarray([[1.0, 0.0]], dtype="float32"), items, {"schema_version": "1.0"})
    original = FaissStore.build

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected build failure")

    monkeypatch.setattr(FaissStore, "build", fail)
    with pytest.raises(RuntimeError, match="injected"):
        activate_faiss_atomically(str(index_dir), np.asarray([[0.0, 1.0]], dtype="float32"), [{"chunk_id": "new", "text": "new"}], {"schema_version": "1.0"})
    monkeypatch.setattr(FaissStore, "build", original)
    store = FaissStore(str(index_dir))
    assert store.load()
    assert store.meta["items"][0]["chunk_id"] == "old"
    with pytest.raises(ValueError, match="align"):
        activate_faiss_atomically(str(tmp_path / "invalid"), np.empty((0, 2), dtype="float32"), [], {})


def test_registry_job_retry_stale_lease_and_events(tmp_path):
    registry = IngestionRegistry(str(tmp_path / "jobs.db"))
    job_id = registry.enqueue_job({"path": "x", "source_name": "x.txt", "options": {}})
    claimed = registry.claim_job(lease_seconds=-1)
    assert claimed["job_id"] == job_id
    assert registry.recover_stale_jobs() == 1
    claimed = registry.claim_job(lease_seconds=30)
    registry.update_job(job_id, "parsing", 0.25, "Parsing", 30)
    registry.fail_job(job_id, "BROKEN", "failed")
    assert registry.get_job(job_id)["events"][-1]["stage"] == "failed"
    assert registry.retry_job(job_id, 3)
    assert not registry.retry_job(job_id, 3)
    assert registry.get_job("missing") is None
    assert registry.list_jobs()[0]["job_id"] == job_id
    assert registry.claim_job(30) is not None
    assert registry.claim_job(30) is None
    cancelled = registry.enqueue_job({"path": "y", "source_name": "y.txt", "options": {}})
    assert registry.cancel_job(cancelled)
    assert registry.get_job(cancelled)["status"] == "cancelled"
    assert not registry.cancel_job(cancelled)
    assert registry.acquire_lock("index", "owner-1", timeout_seconds=0.1)
    assert not registry.acquire_lock("index", "owner-2", timeout_seconds=0.01)
    registry.release_lock("index", "owner-1")
    assert registry.acquire_lock("index", "owner-2", timeout_seconds=0.1)
    registry.release_lock("index", "owner-2")


def test_worker_completes_staged_job_and_records_progress(tmp_path):
    registry, service = make_service(tmp_path)
    job_id = service.stage_upload("worker.md", b"# Worker\nProcess this document.", IngestionOptions(embedding_model="fake"))
    worker = IngestionWorker(service, lease_seconds=10, poll_seconds=0.01)
    worker.start()
    deadline = time.time() + 5
    while time.time() < deadline and registry.get_job(job_id)["status"] not in {"succeeded", "succeeded_with_warnings", "failed"}:
        time.sleep(0.02)
    worker.stop()
    job = registry.get_job(job_id)
    assert job["status"] == "succeeded"
    assert {event["stage"] for event in job["events"]} >= {"queued", "parsing", "ocr", "chunking", "context", "embedding", "validation", "activation", "complete"}


def test_gemini_context_is_cached_and_provider_failure_falls_back(tmp_path):
    class Models:
        def __init__(self, fail=False):
            self.calls, self.fail = 0, fail

        def generate_content(self, **_kwargs):
            self.calls += 1
            if self.fail:
                raise RuntimeError("offline")
            return type("Response", (), {"text": "This chunk belongs to the current operations guide."})()

    class Client:
        def __init__(self, fail=False):
            self.models = Models(fail)

    registry = IngestionRegistry(str(tmp_path / "cached.db"))
    client = Client()
    service = IngestionService(registry, str(tmp_path / "index"), str(tmp_path / "uploads"), FakeEmbedder, client)
    path = tmp_path / "context.md"
    path.write_text("# Operations\nRotate AU-7 credentials.", encoding="utf-8")
    service.process_payload(payload(path, context_mode="gemini"))
    assert client.models.calls == 1
    assert registry.document_chunks(stable_document_id("context.md"))[0]["context_provenance"] == "gemini_cached"

    failing_registry = IngestionRegistry(str(tmp_path / "fallback.db"))
    failing = IngestionService(failing_registry, str(tmp_path / "fallback-index"), str(tmp_path / "fallback-uploads"), FakeEmbedder, Client(True))
    failing.process_payload(payload(path, context_mode="gemini"))
    chunk = failing_registry.document_chunks(stable_document_id("context.md"))[0]
    assert chunk["context_provenance"] == "deterministic"


def test_upload_validation_registry_mismatch_and_empty_delete(tmp_path, monkeypatch):
    registry, service = make_service(tmp_path)
    with pytest.raises(Exception) as error:
        service.stage_upload("bad.csv", b"a,b", IngestionOptions(embedding_model="fake"))
    assert error.value.code == "UNSUPPORTED_FORMAT"
    with pytest.raises(Exception) as error:
        service.stage_upload("fake.pdf", b"not a pdf", IngestionOptions(embedding_model="fake"))
    assert error.value.code == "INVALID_FILE_SIGNATURE"
    with pytest.raises(Exception) as error:
        service.stage_upload("fake.docx", b"not office", IngestionOptions(embedding_model="fake"))
    assert error.value.code == "INVALID_FILE_SIGNATURE"
    import backend.ingestion as ingestion_module
    monkeypatch.setattr(ingestion_module, "SETTINGS", replace(ingestion_module.SETTINGS, max_upload_mb=0))
    with pytest.raises(Exception) as error:
        service.stage_upload("large.txt", b"x", IngestionOptions(embedding_model="fake"))
    assert error.value.code == "FILE_TOO_LARGE"

    path = tmp_path / "only.md"
    path.write_text("# Only\nThe only active document.", encoding="utf-8")
    result = service.process_payload(payload(path))
    stale_store = FaissStore(str(tmp_path / "index"))
    assert stale_store.load()
    assert service.delete_document(result["document_id"], "fake")
    assert not (tmp_path / "index").exists()
    assert not stale_store.load()
    assert stale_store.index is None and stale_store.meta == {"items": []}
    assert registry.active_items_and_vectors("fake")[0] == []


def test_service_restores_registry_when_activation_fails(tmp_path, monkeypatch):
    registry, service = make_service(tmp_path)
    path = tmp_path / "rollback.md"
    path.write_text("# Rollback\nOriginal version.", encoding="utf-8")
    original = service.process_payload(payload(path))
    old_state = registry.document_state(original["document_id"])
    path.write_text("# Rollback\nBroken update.", encoding="utf-8")
    monkeypatch.setattr("backend.ingestion.activate_faiss_atomically", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("activation failed")))
    with pytest.raises(RuntimeError, match="activation failed"):
        service.process_payload(payload(path))
    assert registry.document_state(original["document_id"]) == old_state


def test_clean_rebuild_manifest_is_deterministic_except_timestamp(tmp_path):
    manifests = []
    for root_name in ("first", "second"):
        root = tmp_path / root_name
        root.mkdir()
        _registry, service = make_service(root)
        path = root / "stable.md"
        path.write_text("# Stable\nDeterministic content TS-999.", encoding="utf-8")
        service.process_payload(payload(path))
        manifest = json.loads((root / "index" / "manifest.json").read_text(encoding="utf-8"))
        manifest.pop("built_at")
        manifests.append(manifest)
    assert manifests[0] == manifests[1]
