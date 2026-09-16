import asyncio
from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError
from starlette.datastructures import Headers

import api
from backend.retriever import RetrievalHit, RetrievalResult
from backend.schemas import AskRequest
from backend.review_registry import ReviewRegistry
from backend.schemas import ReviewDecisionRequest


def test_ask_api_accepts_hybrid_rerank_and_returns_stage_fields(monkeypatch):
    api.store.index = object()
    monkeypatch.setattr(api.store, "load", lambda: True)
    monkeypatch.setattr(api, "get_embedder", lambda: object())
    hit = RetrievalHit(
        item={"chunk_id": "c1", "text": "evidence", "source": "demo.md", "page": 1},
        rank=1,
        final_score=2.5,
        dense_rank=2,
        lexical_rank=1,
        fusion_rank=1,
        fusion_score=0.03,
        reranker_rank=1,
        reranker_score=2.5,
        stages=("dense", "lexical", "reranker"),
    )
    monkeypatch.setattr(api, "run_query", lambda **kwargs: {
        "answer": "answer", "citations": [{"chunk_id": "c1"}], "grounding": {},
        "generation": {"mode": "test", "usage": {}},
        "latency_ms": {"retrieval_ms": 1, "generation_ms": 1, "verification_ms": 0, "total_ms": 2},
        "retrieval_result": RetrievalResult("hybrid_rerank", [hit], 50, 8.0, 1.0, 1.0, 1.0, 5.0),
    })
    body = api.ask(
        AskRequest(question="question", top_k=2, retrieval_strategy="hybrid_rerank", rerank_candidates=2)
    )
    assert set(("answer", "citations", "retrieved")) <= body.keys()
    assert body["retrieved"][0]["fusion_rank"] == 1
    assert body["retrieved"][0]["reranker_score"] == 2.5
    assert "grounding" in body


def test_ask_api_rejects_invalid_strategy_and_candidate_bounds():
    with pytest.raises(ValidationError):
        AskRequest(question="q", retrieval_strategy="invalid")
    with pytest.raises(ValidationError):
        AskRequest(question="q", rerank_candidates=0)
    with pytest.raises(ValidationError):
        AskRequest(question="q", retrieval_strategy="lexical", embedding_model="arbitrary/model")


def test_ingestion_api_accepts_multipart_and_validates_bounds(monkeypatch):
    monkeypatch.setattr(api.ingestion_service, "stage_upload", lambda filename, content, options: "job_test")
    monkeypatch.setattr(api.ingestion_worker, "start", lambda: None)
    upload = UploadFile(BytesIO(b"hello"), filename="guide.txt", headers=Headers({"content-type": "text/plain"}))
    response = asyncio.run(api.create_ingestion_jobs(
        files=[upload], context_mode="deterministic", source_version="1.0",
        child_tokens=420, overlap_tokens=80, parent_tokens=1200,
    ))
    assert response["jobs"][0]["job_id"] == "job_test"

    invalid = UploadFile(BytesIO(b"a,b"), filename="data.csv", headers=Headers({"content-type": "text/csv"}))
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.create_ingestion_jobs(
            files=[invalid], context_mode="deterministic", source_version=None,
            child_tokens=420, overlap_tokens=80, parent_tokens=1200,
        ))
    assert error.value.status_code == 415

    too_many = [UploadFile(BytesIO(b"x"), filename=f"{index}.txt", headers=Headers({"content-type": "text/plain"})) for index in range(api.SETTINGS.max_documents_per_job + 1)]
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.create_ingestion_jobs(
            files=too_many, context_mode="deterministic", source_version=None,
            child_tokens=420, overlap_tokens=80, parent_tokens=1200,
        ))
    assert error.value.status_code == 422


def test_ingestion_api_missing_resources_and_delete(monkeypatch):
    monkeypatch.setattr(api.ingestion_registry, "get_job", lambda _job_id: None)
    with pytest.raises(HTTPException) as error:
        api.get_ingestion_job("missing")
    assert error.value.status_code == 404
    monkeypatch.setattr(api.ingestion_service, "delete_document", lambda _document_id: False)
    with pytest.raises(HTTPException) as error:
        api.delete_document("missing")
    assert error.value.status_code == 404
    monkeypatch.setattr(api.ingestion_registry, "cancel_job", lambda _job_id: False)
    with pytest.raises(HTTPException) as error:
        api.cancel_ingestion_job("running")
    assert error.value.status_code == 409


def test_review_api_lists_reads_and_decides(tmp_path, monkeypatch):
    registry = ReviewRegistry(str(tmp_path / "reviews.db"))
    case_id = registry.enqueue({"text": "claim", "verdict": "disputed", "evidence": [{"chunk_id": "c1"}]}, "disputed", "cfg")
    monkeypatch.setattr(api, "review_registry", registry)
    assert api.list_review_cases()["cases"][0]["case_id"] == case_id
    assert api.get_review_case(case_id)["status"] == "open"
    resolved = api.submit_review_decision(case_id, ReviewDecisionRequest(decision="contradicted", reviewer="test"))
    assert resolved["status"] == "resolved"
    with pytest.raises(HTTPException):
        api.get_review_case("missing")
