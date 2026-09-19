from types import SimpleNamespace
import sqlite3

import httpx
import pytest

from backend import demo_budget, query_service
from backend.demo_budget import DemoProviderBudget
from backend.generation_usage import generation_badge, usage_from_response
from backend.grounding import generate_public_exact_draft
from backend.grounding_models import (
    DraftClaim,
    EvidenceSelection,
    PublicEvidenceDraft,
    StructuredDraft,
)
from backend.provider_client import GroqClient, ProviderRequestError
from backend.review_registry import ReviewRegistry
from backend.security_models import RetrievalScope


def item(text="Audit events are retained for 400 days."):
    return {
        "chunk_id": "c1", "text": text, "generation_text": text,
        "retrieval_text": text, "source": "demo.md", "page": 1,
        "organization_id": "org_public", "trust_state": "public",
    }


def test_groq_client_uses_strict_schema_and_sanitizes_failures():
    captured = {}

    def success(request: httpx.Request):
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"answerable":true,"selections":[]}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        })

    client = GroqClient("secret", transport=httpx.MockTransport(success))
    response = client.models.generate_content(
        model="openai/gpt-oss-20b", contents="select evidence",
        config={"response_schema": PublicEvidenceDraft, "max_output_tokens": 64},
    )
    assert response.parsed.answerable is True
    assert response.usage_metadata.total_token_count == 16
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["max_completion_tokens"] == 64
    assert captured["reasoning_effort"] == "low"

    failing = GroqClient(
        "secret",
        transport=httpx.MockTransport(lambda _request: httpx.Response(401, json={"error": "do not expose"})),
    )
    with pytest.raises(ProviderRequestError, match="401:authentication") as caught:
        failing.models.generate_content(model="openai/gpt-oss-20b", contents="private")
    assert "do not expose" not in str(caught.value)
    assert "private" not in str(caught.value)


def test_generation_badges_distinguish_provider_and_quota_fallbacks():
    assert generation_badge({"mode": "groq_assisted_exact_evidence"}) == "Groq-assisted"
    assert generation_badge({"mode": "exact_extractive_fallback"}) == "Exact extractive fallback"
    assert generation_badge({
        "mode": "exact_extractive_fallback", "fallback_reason": "session_day_limit",
    }) == "Quota fallback"
    assert generation_badge({
        "mode": "exact_extractive_fallback", "fallback_reason": "provider_quota",
    }) == "Quota fallback"


def test_public_provider_requires_verbatim_retrieved_evidence():
    exact = PublicEvidenceDraft(answerable=True, selections=[EvidenceSelection(
        chunk_id="c1", sentence_index=0,
    )])
    response = SimpleNamespace(parsed=exact, usage_metadata=SimpleNamespace(
        prompt_token_count=20, candidates_token_count=8, total_token_count=28,
        cached_content_token_count=3,
    ))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_kwargs: response))
    draft, raw = generate_public_exact_draft(
        "How long?", [(1.0, item())], client, "provider-test"
    )
    assert draft.claims[0].provenance == "extractive"
    assert usage_from_response(raw)["total_tokens"] == 28

    unknown_chunk = PublicEvidenceDraft(answerable=True, selections=[EvidenceSelection(
        chunk_id="missing", sentence_index=0,
    )])
    client.models.generate_content = lambda **_kwargs: SimpleNamespace(parsed=unknown_chunk)
    with pytest.raises(ValueError, match="not retrieved"):
        generate_public_exact_draft("How long?", [(1.0, item())], client, "provider-test")

    invalid_index = PublicEvidenceDraft(answerable=True, selections=[EvidenceSelection(
        chunk_id="c1", sentence_index=2,
    )])
    client.models.generate_content = lambda **_kwargs: SimpleNamespace(parsed=invalid_index)
    with pytest.raises(ValueError, match="outside"):
        generate_public_exact_draft(
            "Question?", [(1.0, item("First sentence. Second sentence."))], client, "provider-test"
        )

    two = PublicEvidenceDraft(answerable=True, selections=[
        EvidenceSelection(chunk_id="c1", sentence_index=0),
        EvidenceSelection(chunk_id="c1", sentence_index=1),
    ])
    client.models.generate_content = lambda **_kwargs: SimpleNamespace(parsed=two)
    with pytest.raises(ValueError, match="selection limit"):
        generate_public_exact_draft(
            "Question?", [(1.0, item("One sentence. Two sentence. Three sentence."))],
            client, "provider-test",
            max_claims=1,
        )


def test_demo_budget_enforces_global_and_session_windows(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        demo_provider_global_rpm=2, demo_provider_global_rpd=20,
        demo_provider_session_rpm=1, demo_provider_session_rpd=5,
        demo_provider_circuit_seconds=300,
    )
    monkeypatch.setattr(demo_budget, "SETTINGS", settings)
    budget = DemoProviderBudget(str(tmp_path / "budget.db"))
    assert budget.reserve("session-a", now=1000).allowed
    assert budget.reserve("session-a", now=1001).reason == "session_minute_limit"
    assert budget.reserve("session-b", now=1001).allowed
    assert budget.reserve("session-c", now=1001).reason == "global_minute_limit"
    assert budget.reserve("session-a", now=1061).allowed
    with sqlite3.connect(tmp_path / "budget.db") as connection:
        subjects = {row[0] for row in connection.execute("SELECT subject_hash FROM demo_usage")}
    assert "session-a" not in subjects
    assert any(len(value) == 64 for value in subjects if value != "global")
    budget.open_circuit(now=1100)
    assert budget.reserve("session-z", now=1101).reason == "provider_circuit_open"


def test_demo_budget_daily_reset_and_concurrency(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        demo_provider_global_rpm=100, demo_provider_global_rpd=1,
        demo_provider_session_rpm=100, demo_provider_session_rpd=1,
        demo_provider_circuit_seconds=300,
    )
    monkeypatch.setattr(demo_budget, "SETTINGS", settings)
    budget = DemoProviderBudget(str(tmp_path / "daily.db"))
    assert budget.reserve("session", now=1000).allowed
    assert budget.reserve("session", now=1061).reason == "global_day_limit"
    assert budget.reserve("session", now=86401).allowed
    assert budget.concurrency.acquire(blocking=False)
    assert not budget.concurrency.acquire(blocking=False)
    budget.concurrency.release()


def test_low_memory_query_forces_lexical_and_abstains_on_unknown(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        low_memory_demo=True, demo_question_max_chars=500, demo_top_k_max=6,
        public_generation_enabled=False, genai_pricing_tier="free",
        generation_provider="groq", generation_model="provider-test",
        demo_context_max_chars=12000, demo_provider_timeout_seconds=12,
    )
    monkeypatch.setattr(query_service, "SETTINGS", settings)
    monkeypatch.setattr(query_service, "log_run", lambda _row: None)
    store = SimpleNamespace(meta={"items": [item()]})
    scope = RetrievalScope("org_public", "anonymous_demo", "test")
    reviews = ReviewRegistry(str(tmp_path / "reviews.db"))
    result = query_service.run_query(
        store=store, question="How long are audit events retained?", top_k=3,
        strategy="lexical", scope=scope, review_registry=reviews,
        client_key="session", organization_id="org_public",
        embedder_factory=lambda: (_ for _ in ()).throw(AssertionError("model loaded")),
    )
    assert "400 days" in result["answer"]
    assert result["generation"]["fallback_used"]
    assert result["grounding"]["verifier"]["model"] == "deterministic-exact-evidence"

    unknown = query_service.run_query(
        store=store, question="What is the manufacturing cost of the hardware appliance?", top_k=3,
        strategy="lexical", scope=scope, review_registry=reviews,
        client_key="session", organization_id="org_public",
    )
    assert unknown["answer"].startswith("I don't know")

    with pytest.raises(query_service.PublicDemoPolicyError, match="lexical"):
        query_service.run_query(
            store=store, question="question", top_k=3, strategy="dense",
            scope=scope, review_registry=reviews, client_key="session",
        )
    with pytest.raises(query_service.PublicDemoPolicyError, match="top_k"):
        query_service.run_query(
            store=store, question="question", top_k=7, strategy="lexical",
            scope=scope, review_registry=reviews, client_key="session",
        )
    with pytest.raises(query_service.PublicDemoPolicyError, match="500"):
        query_service.run_query(
            store=store, question="x" * 501, top_k=3, strategy="lexical",
            scope=scope, review_registry=reviews, client_key="session",
        )


def test_public_query_provider_success_and_invalid_response_fallback(tmp_path, monkeypatch):
    query_settings = SimpleNamespace(
        low_memory_demo=True, demo_question_max_chars=500, demo_top_k_max=6,
        public_generation_enabled=True, genai_pricing_tier="free",
        generation_provider="groq", generation_model="provider-test",
        demo_context_max_chars=12000, demo_provider_timeout_seconds=12,
    )
    budget_settings = SimpleNamespace(
        demo_provider_global_rpm=100, demo_provider_global_rpd=100,
        demo_provider_session_rpm=100, demo_provider_session_rpd=100,
        demo_provider_circuit_seconds=300,
    )
    monkeypatch.setattr(query_service, "SETTINGS", query_settings)
    monkeypatch.setattr(demo_budget, "SETTINGS", budget_settings)
    monkeypatch.setattr(query_service, "log_run", lambda _row: None)
    budget = DemoProviderBudget(str(tmp_path / "provider-success.db"))
    store = SimpleNamespace(meta={"items": [item()]})
    scope = RetrievalScope("org_public", "anonymous_demo", "test")
    reviews = ReviewRegistry(str(tmp_path / "reviews.db"))
    exact = PublicEvidenceDraft(answerable=True, selections=[EvidenceSelection(
        chunk_id="c1", sentence_index=0,
    )])
    response = SimpleNamespace(parsed=exact, usage_metadata=SimpleNamespace(
        prompt_token_count=20, candidates_token_count=8, total_token_count=28,
        cached_content_token_count=0,
    ))
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_kwargs: response))
    result = query_service.run_query(
        store=store, question="How long are audit events retained?", top_k=3,
        strategy="lexical", scope=scope, review_registry=reviews,
        client_key="success", generation_client=client, budget=budget,
    )
    assert result["generation"]["mode"] == "groq_assisted_exact_evidence"
    assert result["generation"]["usage"]["total_tokens"] == 28

    invalid = PublicEvidenceDraft(answerable=True, selections=[EvidenceSelection(
        chunk_id="c1", sentence_index=99,
    )])
    invalid_client = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **_kwargs: SimpleNamespace(parsed=invalid)
    ))
    invalid_budget = DemoProviderBudget(str(tmp_path / "invalid.db"))
    fallback = query_service.run_query(
        store=store, question="How long are audit events retained?", top_k=3,
        strategy="lexical", scope=scope, review_registry=reviews,
        client_key="invalid", generation_client=invalid_client,
        budget=invalid_budget,
    )
    assert fallback["generation"]["mode"] == "exact_extractive_fallback"
    assert fallback["generation"]["fallback_reason"] == "invalid_provider_response"
    assert invalid_budget.reserve("after-invalid").allowed


@pytest.mark.parametrize(("error", "expected"), [
    (query_service.FutureTimeout(), "provider_timeout"),
    (RuntimeError("429 quota exceeded"), "provider_quota"),
    (ProviderRequestError(401, "authentication"), "provider_authentication"),
    (ProviderRequestError(404, "model_not_found"), "provider_model_not_found"),
])
def test_public_query_provider_failures_fall_back(tmp_path, monkeypatch, error, expected):
    query_settings = SimpleNamespace(
        low_memory_demo=True, demo_question_max_chars=500, demo_top_k_max=6,
        public_generation_enabled=True, genai_pricing_tier="free",
        generation_provider="groq", generation_model="provider-test",
        demo_context_max_chars=12000, demo_provider_timeout_seconds=12,
    )
    budget_settings = SimpleNamespace(
        demo_provider_global_rpm=100, demo_provider_global_rpd=100,
        demo_provider_session_rpm=100, demo_provider_session_rpd=100,
        demo_provider_circuit_seconds=300,
    )
    monkeypatch.setattr(query_service, "SETTINGS", query_settings)
    monkeypatch.setattr(demo_budget, "SETTINGS", budget_settings)
    monkeypatch.setattr(query_service, "log_run", lambda _row: None)
    budget = DemoProviderBudget(str(tmp_path / "provider.db"))

    def fail(*_args, **_kwargs):
        budget.concurrency.release()
        raise error

    monkeypatch.setattr(query_service, "_public_draft_with_timeout", fail)
    result = query_service.run_query(
        store=SimpleNamespace(meta={"items": [item()]}),
        question="How long are audit events retained?", top_k=3, strategy="lexical",
        scope=RetrievalScope("org_public", "anonymous_demo", "test"),
        review_registry=ReviewRegistry(str(tmp_path / "reviews.db")),
        client_key="provider", generation_client=SimpleNamespace(), budget=budget,
    )
    assert result["generation"]["fallback_reason"] == expected
    assert "400 days" in result["answer"]
