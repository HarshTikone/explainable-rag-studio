from typing import Dict, Any
import time

from .prompt import build_context
from .retriever import RetrievalResult
from .grounding import (
    build_extractive_draft,
    citations_from_grounding,
    generate_structured_draft_with_response,
    grounding_config_fingerprint,
    render_grounded_answer,
    verify_claims,
)
from .review_registry import ReviewRegistry
from .config import SETTINGS
from .grounding_policy import GroundingPolicy
from .grounding_models import StructuredDraft
from .generation_usage import usage_from_response
from .observability import set_span_attributes, span

def _legacy_items(retrieved_items):
    return retrieved_items.as_legacy() if isinstance(retrieved_items, RetrievalResult) else retrieved_items


def extractive_answer(question: str, retrieved_items) -> str:
    """
    Non-LLM fallback: returns the most relevant chunk excerpt.
    Keeps the app usable even without an API key.
    """
    draft = build_extractive_draft(retrieved_items, question=question)
    return " ".join(claim.text for claim in draft.claims) or "I don't know."

def answer_with_optional_llm(
    question: str,
    retrieved_items,
    use_gemini: bool,
    gemini_client=None,
    gemini_model: str = "",
    verifier=None,
    review_registry: ReviewRegistry | None = None,
    persist_review: bool = True,
    grounding_policy: GroundingPolicy | None = None,
    extractive_max_claims: int = 3,
    draft_override: StructuredDraft | None = None,
    semantic_scorer=None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        answer: str,
        citations: [{chunk_id, source, page, score}],
        context: str
      }
    """
    generation_started = time.perf_counter()
    legacy = _legacy_items(retrieved_items)
    retrieved = [it for _, it in legacy]
    context = build_context(retrieved)
    generation = {
        "mode": "exact_extractive_fallback", "provider": "local",
        "model": "deterministic-extractive", "fallback_used": True,
        "fallback_reason": "gemini_not_requested",
        "usage": usage_from_response(None),
    }

    if draft_override is not None:
        draft = draft_override
        generation["fallback_reason"] = "draft_override"
    elif not legacy:
        draft = build_extractive_draft([], max_claims=extractive_max_claims, question=question)
    elif use_gemini and gemini_client is not None:
        try:
            with span("rag.generate", {
                "gen_ai.system": "google", "gen_ai.request.model": gemini_model,
                "rag.generation.mode": "structured",
            }) as generation_span:
                draft, response = generate_structured_draft_with_response(
                    question, retrieved_items, gemini_client, gemini_model
                )
                usage = usage_from_response(response)
                set_span_attributes(generation_span, {
                    "gen_ai.usage.input_tokens": usage["input_tokens"],
                    "gen_ai.usage.output_tokens": usage["output_tokens"],
                    "gen_ai.usage.cached_tokens": usage["cached_tokens"],
                })
            generation = {
                "mode": "gemini_structured", "provider": "google", "model": gemini_model,
                "fallback_used": False, "fallback_reason": "",
                "usage": usage,
            }
        except Exception:
            draft = build_extractive_draft(retrieved_items, max_claims=extractive_max_claims, question=question)
            generation["fallback_reason"] = "provider_or_validation_error"
    else:
        draft = build_extractive_draft(retrieved_items, max_claims=extractive_max_claims, question=question)
    claim_generation_ms = (time.perf_counter() - generation_started) * 1000

    with span("rag.verify", {"rag.verifier.mode": getattr(verifier, "model_name", "default")}):
        verification = verify_claims(
            draft, retrieved_items, verifier=verifier, policy=grounding_policy,
            semantic_scorer=semantic_scorer,
        ).result
    verification.latency_ms["claim_generation"] = claim_generation_ms
    verification.latency_ms.setdefault("citation_validation", 0.0)
    verification.latency_ms.setdefault("conflict_scan", 0.0)
    answer = render_grounded_answer(verification)
    citations = citations_from_grounding(verification)

    if persist_review:
        registry = review_registry or ReviewRegistry(SETTINGS.review_db_path)
        fingerprint = grounding_config_fingerprint(verification)
        for claim in verification.accepted_claims + verification.rejected_claims:
            if claim.low_confidence or claim.verdict == "disputed":
                registry.enqueue(
                    claim.model_dump(),
                    "disputed" if claim.verdict == "disputed" else "low_confidence",
                    fingerprint,
                )

    return {
        "answer": answer,
        "citations": citations,
        "context": context,
        "grounding": verification.model_dump(),
        "generation": generation,
    }
