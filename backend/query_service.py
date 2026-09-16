"""One query path shared by FastAPI and Streamlit."""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable, Dict

from .config import SETTINGS
from .demo_budget import DEMO_GEMINI_BUDGET, DemoGeminiBudget
from .generation_usage import usage_from_response
from .grounding import DeterministicOnlyVerifier, EmbeddingSemanticScorer, generate_public_exact_draft
from .observability import set_span_attributes, span
from .qa import answer_with_optional_llm
from .retriever import retrieve
from .security_models import RetrievalScope
from .telemetry import log_run
from .utils import now_ms


class PublicDemoPolicyError(ValueError):
    pass


_GEMINI_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="public-gemini")


def create_gemini_client():
    """Create the configured provider client with the public wall-clock timeout."""
    if not SETTINGS.gemini_api_key.strip():
        return None
    from google import genai
    options = None
    if SETTINGS.low_memory_demo:
        options = {"timeout": int(SETTINGS.demo_gemini_timeout_seconds * 1000)}
    return genai.Client(api_key=SETTINGS.gemini_api_key, http_options=options)


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


def _public_draft_with_timeout(question, found, client, budget: DemoGeminiBudget):
    try:
        future = _GEMINI_EXECUTOR.submit(
            generate_public_exact_draft,
            question,
            found,
            client,
            SETTINGS.gemini_model,
            context_max_chars=SETTINGS.demo_context_max_chars,
            max_claims=2,
        )
    except Exception:
        budget.concurrency.release()
        raise
    future.add_done_callback(lambda _future: budget.concurrency.release())
    return future.result(timeout=SETTINGS.demo_gemini_timeout_seconds)


def _provider_fallback_reason(exc: Exception) -> str:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    message = str(exc).casefold()
    if status == 429 or "429" in message or "quota" in message or "resource exhausted" in message:
        return "provider_quota"
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
    gemini_client=None,
    embedder_factory: Callable[[], Any] | None = None,
    budget: DemoGeminiBudget = DEMO_GEMINI_BUDGET,
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
        generation = _fallback_metadata("gemini_not_configured")
        should_try_public_gemini = (
            SETTINGS.low_memory_demo
            and SETTINGS.public_gemini_enabled
            and gemini_client is not None
        )
        if should_try_public_gemini:
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
                            "gen_ai.system": "google",
                            "gen_ai.request.model": SETTINGS.gemini_model,
                            "rag.generation.mode": "public_exact",
                        }) as generation_span:
                            draft, response = _public_draft_with_timeout(
                                normalized_question, found, gemini_client, budget
                            )
                            usage = usage_from_response(response)
                            set_span_attributes(generation_span, {
                                "gen_ai.usage.input_tokens": usage["input_tokens"],
                                "gen_ai.usage.output_tokens": usage["output_tokens"],
                                "gen_ai.usage.cached_tokens": usage["cached_tokens"],
                            })
                        budget.reset_circuit()
                        generation = {
                            "mode": "gemini_assisted_exact_evidence",
                            "provider": "google",
                            "model": SETTINGS.gemini_model,
                            "fallback_used": False,
                            "fallback_reason": "",
                            "usage": usage,
                        }
                    except FutureTimeout:
                        budget.open_circuit()
                        generation = _fallback_metadata("provider_timeout")
                    except Exception as exc:
                        budget.open_circuit()
                        generation = _fallback_metadata(_provider_fallback_reason(exc))

        verifier = DeterministicOnlyVerifier() if SETTINGS.low_memory_demo else None
        semantic_scorer = None if SETTINGS.low_memory_demo or embedder is None else EmbeddingSemanticScorer(embedder)
        output = answer_with_optional_llm(
            question=normalized_question,
            retrieved_items=found,
            use_gemini=bool(gemini_client) and not SETTINGS.low_memory_demo,
            gemini_client=gemini_client,
            gemini_model=SETTINGS.gemini_model,
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
