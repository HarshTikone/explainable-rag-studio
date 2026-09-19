"""One query path shared by FastAPI and Streamlit."""
from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, Dict

from .config import SETTINGS
from .demo_budget import DEMO_PROVIDER_BUDGET, DemoProviderBudget
from .generation_usage import usage_from_response
from .grounding import (
    DeterministicOnlyVerifier,
    EmbeddingSemanticScorer,
    PublicDraftValidationError,
    generate_public_exact_draft,
)
from .observability import set_span_attributes, span
from .provider_client import GroqClient
from .qa import answer_with_optional_llm
from .retriever import retrieve
from .security_models import RetrievalScope
from .telemetry import log_run
from .utils import now_ms


class PublicDemoPolicyError(ValueError):
    pass


LOGGER = logging.getLogger(__name__)
_PROVIDER_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="public-generation")


def create_generation_client():
    """Create the configured provider client with the public wall-clock timeout."""
    if not SETTINGS.groq_api_key.strip():
        return None
    if SETTINGS.generation_provider != "groq":
        raise ValueError(f"Unsupported generation provider: {SETTINGS.generation_provider}")
    return GroqClient(
        api_key=SETTINGS.groq_api_key,
        timeout_seconds=SETTINGS.demo_provider_timeout_seconds,
    )


def _fallback_metadata(reason: str) -> Dict[str, Any]:
    return {
        "mode": "exact_extractive_fallback",
        "provider": "local",
        "model": "deterministic-extractive",
        "fallback_used": True,
        "fallback_reason": reason,
        "usage": {
            "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
            "total_tokens": 0, "estimated_cost_usd": 0.0,
            "pricing_tier": SETTINGS.genai_pricing_tier,
        },
    }


def _public_draft_with_timeout(question, found, client, budget: DemoProviderBudget):
    try:
        future = _PROVIDER_EXECUTOR.submit(
            generate_public_exact_draft,
            question,
            found,
            client,
            SETTINGS.generation_model,
            context_max_chars=SETTINGS.demo_context_max_chars,
            max_claims=2,
        )
    except Exception:
        budget.concurrency.release()
        raise
    future.add_done_callback(lambda _future: budget.concurrency.release())
    return future.result(timeout=SETTINGS.demo_provider_timeout_seconds)


def _provider_fallback_reason(exc: Exception) -> str:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    message = str(exc).casefold()
    if status == 429 or "429" in message or "quota" in message or "resource exhausted" in message:
        return "provider_quota"
    if status == 400:
        return "provider_invalid_request"
    if status == 401:
        return "provider_authentication"
    if status == 403:
        return "provider_permission"
    if status == 404:
        return "provider_model_not_found"
    if isinstance(status, int) and status >= 500:
        return "provider_unavailable"
    return "provider_or_validation_error"


def run_query(
    *,
    store,
    question: str,
    top_k: int,
    strategy: str,
    scope: RetrievalScope,
    review_registry,
    client_key: str,
    generation_client=None,
    embedder_factory: Callable[[], Any] | None = None,
    budget: DemoProviderBudget = DEMO_PROVIDER_BUDGET,
    organization_id: str = "org_public",
    actor_user_id: str = "",
    rerank_candidates: int | None = None,
) -> Dict[str, Any]:
    normalized_question = " ".join((question or "").split())
    if not normalized_question:
        raise PublicDemoPolicyError("Question cannot be empty.")
    if SETTINGS.low_memory_demo:
        if len(normalized_question) > SETTINGS.demo_question_max_chars:
            raise PublicDemoPolicyError(
                f"Public demo questions are limited to {SETTINGS.demo_question_max_chars} characters."
            )
        if strategy != "lexical":
            raise PublicDemoPolicyError("The public free-tier demo supports lexical retrieval only.")
        if not 1 <= top_k <= SETTINGS.demo_top_k_max:
            raise PublicDemoPolicyError(
                f"The public free-tier demo supports top_k from 1 to {SETTINGS.demo_top_k_max}."
            )
    query_hash = hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()
    started = time.perf_counter()
    with span("rag.query", {
        "rag.profile": "public_low_memory" if SETTINGS.low_memory_demo else "full",
        "rag.query.sha256": query_hash,
        "rag.query.length": len(normalized_question),
        "rag.retrieval.strategy": strategy,
        "rag.retrieval.top_k": top_k,
    }) as query_span:
        embedder = None
        if strategy != "lexical":
            if embedder_factory is None:
                raise PublicDemoPolicyError("An embedder is required for this retrieval strategy.")
            embedder = embedder_factory()
        with span("rag.retrieve", {"rag.retrieval.strategy": strategy, "rag.retrieval.top_k": top_k}):
            found = retrieve(
                store=store,
                embedder=embedder,
                query=normalized_question,
                top_k=top_k,
                strategy=strategy,
                rerank_candidates=rerank_candidates,
                scope=scope,
            )
        retrieved_at = time.perf_counter()

        draft = None
        generation = _fallback_metadata("provider_not_configured")
        should_try_public_provider = (
            SETTINGS.low_memory_demo
            and SETTINGS.public_generation_enabled
            and generation_client is not None
        )
        if should_try_public_provider:
            if not budget.concurrency.acquire(blocking=False):
                generation = _fallback_metadata("concurrency_limit")
            else:
                decision = budget.reserve(client_key)
                if not decision.allowed:
                    budget.concurrency.release()
                    generation = _fallback_metadata(decision.reason)
                else:
                    try:
                        with span("rag.generate", {
                            "gen_ai.system": SETTINGS.generation_provider,
                            "gen_ai.request.model": SETTINGS.generation_model,
                            "rag.generation.mode": "public_exact",
                        }) as generation_span:
                            draft, response = _public_draft_with_timeout(
                                normalized_question, found, generation_client, budget
                            )
                            usage = usage_from_response(response)
                            set_span_attributes(generation_span, {
                                "gen_ai.usage.input_tokens": usage["input_tokens"],
                                "gen_ai.usage.output_tokens": usage["output_tokens"],
                                "gen_ai.usage.cached_tokens": usage["cached_tokens"],
                            })
                        budget.reset_circuit()
                        generation = {
                            "mode": "groq_assisted_exact_evidence",
                            "provider": SETTINGS.generation_provider,
                            "model": SETTINGS.generation_model,
                            "fallback_used": False,
                            "fallback_reason": "",
                            "usage": usage,
                        }
                    except PublicDraftValidationError:
                        generation = _fallback_metadata("invalid_provider_response")
                    except FutureTimeout:
                        budget.open_circuit()
                        generation = _fallback_metadata("provider_timeout")
                    except Exception as exc:
                        budget.open_circuit()
                        reason = _provider_fallback_reason(exc)
                        LOGGER.warning(
                            "External generation failed: reason=%s status=%s type=%s",
                            reason,
                            getattr(exc, "status_code", None) or getattr(exc, "code", None),
                            type(exc).__name__,
                        )
                        generation = _fallback_metadata(reason)

        verifier = DeterministicOnlyVerifier() if SETTINGS.low_memory_demo else None
        semantic_scorer = None if SETTINGS.low_memory_demo or embedder is None else EmbeddingSemanticScorer(embedder)
        output = answer_with_optional_llm(
            question=normalized_question,
            retrieved_items=found,
            use_provider=bool(generation_client) and not SETTINGS.low_memory_demo,
            generation_client=generation_client,
            generation_model=SETTINGS.generation_model,
            generation_provider=SETTINGS.generation_provider,
            verifier=verifier,
            review_registry=review_registry,
            extractive_max_claims=1 if SETTINGS.low_memory_demo else 3,
            draft_override=draft,
            semantic_scorer=semantic_scorer,
        )
        if not SETTINGS.low_memory_demo:
            generation = output.get("generation", generation)
        finished = time.perf_counter()
        grounding = output.get("grounding", {})
        timings = {
            "retrieval_ms": int((retrieved_at - started) * 1000),
            "generation_ms": int((finished - retrieved_at) * 1000),
            "verification_ms": int(grounding.get("latency_ms", {}).get("total_verification", 0)),
            "total_ms": int((finished - started) * 1000),
        }
        set_span_attributes(query_span, {
            "rag.retrieval.returned_count": len(found.hits),
            "rag.generation.mode": generation["mode"],
            "rag.generation.fallback": generation["fallback_used"],
            "rag.grounding.status": grounding.get("status", ""),
            "rag.latency.total_ms": timings["total_ms"],
        })
        log_run({
            "ts_ms": now_ms(), "query": normalized_question,
            "organization_id": organization_id, "actor_user_id": actor_user_id,
            "top_k": top_k, "use_mmr": strategy == "dense_mmr",
            "retrieval_ms": timings["retrieval_ms"],
            "generation_ms": timings["generation_ms"], "total_ms": timings["total_ms"],
            "citation_count": len(output.get("citations", [])),
            "verification_ms": timings["verification_ms"],
            "grounding_status": grounding.get("status", ""),
            "accepted_claims": len(grounding.get("accepted_claims", [])),
            "rejected_claims": len(grounding.get("rejected_claims", [])),
            "conflict_count": sum(
                1 for claim in grounding.get("rejected_claims", [])
                if claim.get("verdict") in {"contradicted", "disputed"}
            ),
            "generation_mode": generation["mode"],
            "generation_fallback_reason": generation["fallback_reason"],
            **generation["usage"],
        })
        return {
            **output,
            "retrieval_result": found,
            "generation": generation,
            "latency_ms": timings,
        }
