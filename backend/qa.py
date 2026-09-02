from typing import Dict, Any
import time

from .prompt import build_context
from .retriever import RetrievalResult
from .grounding import (
    build_extractive_draft,
    citations_from_grounding,
    generate_structured_draft,
    grounding_config_fingerprint,
    render_grounded_answer,
    verify_claims,
)
from .review_registry import ReviewRegistry
from .config import SETTINGS
from .grounding_policy import GroundingPolicy

def _legacy_items(retrieved_items):
    return retrieved_items.as_legacy() if isinstance(retrieved_items, RetrievalResult) else retrieved_items


def extractive_answer(question: str, retrieved_items) -> str:
    """
    Non-LLM fallback: returns the most relevant chunk excerpt.
    Keeps the app usable even without an API key.
    """
    draft = build_extractive_draft(retrieved_items)
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

    if not legacy:
        draft = build_extractive_draft([])
    elif use_gemini and gemini_client is not None:
        try:
            draft = generate_structured_draft(question, retrieved_items, gemini_client, gemini_model)
        except Exception:
            draft = build_extractive_draft(retrieved_items)
    else:
        draft = build_extractive_draft(retrieved_items)
    claim_generation_ms = (time.perf_counter() - generation_started) * 1000

    verification = verify_claims(draft, retrieved_items, verifier=verifier, policy=grounding_policy).result
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
    }
