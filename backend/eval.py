"""Deterministic, versionable evaluation utilities for RAG experiments."""
from __future__ import annotations

import json
import os
import math
from typing import Any, Dict, Iterable, List, Sequence

from .utils import ensure_dir, write_json


def load_eval_set(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    validate_eval_set(items)
    return items


def validate_eval_set(items: Any) -> None:
    if not isinstance(items, list) or not items:
        raise ValueError("Evaluation set must be a non-empty JSON list.")
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            raise ValueError(f"Item {index + 1} must contain a non-empty question.")
        if item.get("relevant_chunk_ids") is not None and not isinstance(item.get("relevant_chunk_ids"), list):
            raise ValueError(f"Item {index + 1}: relevant_chunk_ids must be a list.")


def simple_accuracy(pred: str, expected: str) -> float:
    prediction = (pred or "").casefold()
    reference = (expected or "").casefold().strip()
    return 1.0 if reference and reference in prediction else 0.0


def _unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def retrieval_metrics(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> Dict[str, float | None]:
    """Compute retrieval metrics; None indicates that gold retrieval labels were not provided."""
    retrieved = _unique(retrieved_ids)
    relevant = set(_unique(relevant_ids))
    if not relevant:
        return {"hit_rate": None, "recall": None, "precision": None, "reciprocal_rank": None, "ndcg": None}
    hits = [chunk_id for chunk_id in retrieved if chunk_id in relevant]
    first_rank = next((rank for rank, chunk_id in enumerate(retrieved, 1) if chunk_id in relevant), None)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, chunk_id in enumerate(retrieved, 1) if chunk_id in relevant)
    ideal_hits = min(len(relevant), len(retrieved))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "hit_rate": 1.0 if hits else 0.0,
        "recall": len(set(hits)) / len(relevant),
        "precision": len(hits) / max(1, len(retrieved)),
        "reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def citation_metrics(
    citations: Sequence[Dict[str, Any]], retrieved_ids: Sequence[str], *, allow_empty: bool = False
) -> Dict[str, float]:
    cited = _unique(citation.get("chunk_id", "") for citation in citations)
    retrieved = set(_unique(retrieved_ids))
    valid = sum(1 for chunk_id in cited if chunk_id in retrieved)
    required_count = len(cited) if cited else (0 if allow_empty else 1)
    validity = valid / required_count if required_count else 1.0
    return {
        "citation_validity": validity,
        "citation_count": float(len(cited)),
        "valid_citation_count": float(valid),
        "citation_denominator": float(required_count),
    }


def is_abstention(answer: str) -> bool:
    normalized = " ".join((answer or "").casefold().split())
    markers = ("i don't know", "i do not know", "not in the context", "insufficient context", "cannot answer")
    return any(marker in normalized for marker in markers)


def _mean_present(results: Sequence[Dict[str, Any]], key: str) -> float | None:
    values = [result[key] for result in results if result.get(key) is not None]
    return (sum(values) / len(values)) if values else None


def percentile(values: Sequence[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run_eval(eval_items: List[Dict[str, Any]], ask_fn, out_dir: str, write_report: bool = True) -> Dict[str, Any]:
    """Run answer, retrieval, citation, and abstention checks in one benchmark."""
    validate_eval_set(eval_items)
    ensure_dir(out_dir)
    results: List[Dict[str, Any]] = []
    for item in eval_items:
        output = ask_fn(item["question"])
        answer = output.get("answer", "")
        citations = output.get("citations", [])
        retrieved = output.get("retrieved", [])
        retrieved_ids = [entry.get("chunk_id", "") if isinstance(entry, dict) else str(entry) for entry in retrieved]
        answerable = bool(item.get("answerable", True))
        abstained = is_abstention(answer)
        metrics_at_k = retrieval_metrics(retrieved_ids, item.get("relevant_chunk_ids", []))
        metrics_at_5 = retrieval_metrics(retrieved_ids[:5], item.get("relevant_chunk_ids", []))
        citation_result = citation_metrics(citations, retrieved_ids, allow_empty=abstained)
        result = {
            "question": item["question"],
            "category": item.get("category", "general"),
            "expected": item.get("expected", item.get("reference_answer", "")),
            "answerable": answerable,
            "expected_source_versions": item.get("expected_source_versions", []),
            "answer": answer,
            "citations": citations,
            "retrieved_chunk_ids": retrieved_ids,
            "retrieval_hits": retrieved,
            "score": simple_accuracy(answer, item.get("expected", item.get("reference_answer", ""))),
            "abstained": abstained,
            "abstention_correct": (not answerable and abstained) or (answerable and not abstained),
            "retrieval_latency_ms": float(output.get("retrieval_latency_ms", 0.0)),
            "retrieval_latency_samples_ms": [float(value) for value in output.get("retrieval_latency_samples_ms", [])],
            "total_latency_ms": float(output.get("total_latency_ms", 0.0)),
            "total_latency_samples_ms": [float(value) for value in output.get("total_latency_samples_ms", [])],
            "stage_latency_ms": output.get("stage_latency_ms", {}),
            "stage_latency_samples_ms": output.get("stage_latency_samples_ms", {}),
            "reranking_trace": output.get("reranking_trace", []),
            "grounding": output.get("grounding", {}),
            **metrics_at_k,
            "recall_at_5": metrics_at_5["recall"],
            "ndcg_at_5": metrics_at_5["ndcg"],
            **citation_result,
        }
        grounding = result["grounding"]
        accepted_claims = grounding.get("accepted_claims", [])
        rejected_claims = grounding.get("rejected_claims", [])
        grounding_labeled = bool(grounding)
        valid_retrieved = set(retrieved_ids)
        supported_claims = [claim for claim in accepted_claims if claim.get("verdict") == "supported"]
        cited_supported_claims = [
            claim for claim in accepted_claims
            if claim.get("cited_chunk_ids")
            and all(chunk_id in valid_retrieved for chunk_id in claim.get("cited_chunk_ids", []))
        ]
        result.update({
            "grounding_status": grounding.get("status"),
            "accepted_claim_count": len(accepted_claims),
            "rejected_claim_count": len(rejected_claims),
            "supported_displayed_claim_count": len(supported_claims),
            "cited_displayed_claim_count": len(cited_supported_claims),
            "displayed_claim_support": (len(supported_claims) / len(accepted_claims)) if accepted_claims else None,
            "claim_citation_coverage": (len(cited_supported_claims) / len(accepted_claims)) if accepted_claims else None,
            "answer_coverage": (1.0 if accepted_claims else 0.0) if grounding_labeled and answerable else None,
            "conflict_count": sum(1 for claim in rejected_claims if claim.get("verdict") == "disputed"),
            "verification_latency_ms": float(grounding.get("latency_ms", {}).get("total_verification", 0.0)),
            "verification_latency_samples_ms": [float(value) for value in output.get("verification_latency_samples_ms", [])],
        })
        results.append(result)

    report = {
        "schema_version": "2.0",
        "n": len(results),
        "accuracy": _mean_present(results, "score") or 0.0,
        "retrieval": {
            "hit_rate": _mean_present(results, "hit_rate"),
            "recall": _mean_present(results, "recall"),
            "precision": _mean_present(results, "precision"),
            "mrr": _mean_present(results, "reciprocal_rank"),
            "ndcg": _mean_present(results, "ndcg"),
            "recall_at_5": _mean_present(results, "recall_at_5"),
            "ndcg_at_5": _mean_present(results, "ndcg_at_5"),
        },
        "citation_validity": (
            sum(result["valid_citation_count"] for result in results)
            / sum(result["citation_denominator"] for result in results)
            if sum(result["citation_denominator"] for result in results) else 1.0
        ),
        "abstention_accuracy": _mean_present(results, "abstention_correct") or 0.0,
        "grounding": {
            "displayed_claim_support": (
                sum(result["supported_displayed_claim_count"] for result in results)
                / sum(result["accepted_claim_count"] for result in results)
                if sum(result["accepted_claim_count"] for result in results) else None
            ),
            "claim_citation_coverage": (
                sum(result["cited_displayed_claim_count"] for result in results)
                / sum(result["accepted_claim_count"] for result in results)
                if sum(result["accepted_claim_count"] for result in results) else None
            ),
            "answer_coverage": _mean_present(results, "answer_coverage"),
            "accepted_claims": sum(result["accepted_claim_count"] for result in results),
            "rejected_claims": sum(result["rejected_claim_count"] for result in results),
            "conflicts": sum(result["conflict_count"] for result in results),
        },
        "latency_ms": {
            "retrieval_p50": percentile([
                value for result in results for value in (result["retrieval_latency_samples_ms"] or [result["retrieval_latency_ms"]])
            ], 0.50),
            "retrieval_p95": percentile([
                value for result in results for value in (result["retrieval_latency_samples_ms"] or [result["retrieval_latency_ms"]])
            ], 0.95),
            "total_p50": percentile([
                value for result in results for value in (result["total_latency_samples_ms"] or [result["total_latency_ms"]])
            ], 0.50),
            "total_p95": percentile([
                value for result in results for value in (result["total_latency_samples_ms"] or [result["total_latency_ms"]])
            ], 0.95),
            "verification_p50": percentile([
                value for result in results for value in (result["verification_latency_samples_ms"] or [result["verification_latency_ms"]])
            ], 0.50),
            "verification_p95": percentile([
                value for result in results for value in (result["verification_latency_samples_ms"] or [result["verification_latency_ms"]])
            ], 0.95),
        },
        "stage_latency_ms": {
            stage: {
                "p50": percentile([
                    value for result in results
                    for value in (result["stage_latency_samples_ms"].get(stage, []) or [result["stage_latency_ms"].get(stage, 0.0)])
                ], 0.50),
                "p95": percentile([
                    value for result in results
                    for value in (result["stage_latency_samples_ms"].get(stage, []) or [result["stage_latency_ms"].get(stage, 0.0)])
                ], 0.95),
            }
            for stage in ("dense", "lexical", "fusion", "reranking")
        },
        "results": results,
    }
    if write_report:
        write_json(os.path.join(out_dir, "eval_report.json"), report)
    return report
