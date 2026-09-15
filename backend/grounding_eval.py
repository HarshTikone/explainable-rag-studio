"""Reproducible claim-verification benchmark and strict-policy promotion gates."""
from __future__ import annotations

import json
import hashlib
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Sequence, Tuple

from .eval import percentile
from .grounding import select_evidence_premise, verify_claims
from .grounding_models import DraftClaim, StructuredDraft
from .grounding_policy import GroundingPolicy, default_grounding_policy
from .utils import ensure_dir, write_json


LABELS = ("supported", "neutral", "contradiction")
GROUNDING_ARTIFACT_SCHEMA_VERSION = "1.1"


def validate_grounding_benchmark(cases: Any, corpus_ids: Sequence[str] | None = None) -> None:
    if not isinstance(cases, list) or len(cases) < 1:
        raise ValueError("Grounding benchmark must be a non-empty list.")
    known = set(corpus_ids or [])
    for index, case in enumerate(cases):
        if not str(case.get("claim", "")).strip():
            raise ValueError(f"Case {index + 1} requires a claim.")
        if case.get("label") not in LABELS:
            raise ValueError(f"Case {index + 1} has an invalid label.")
        chunk_ids = case.get("evidence_chunk_ids")
        if not isinstance(chunk_ids, list) or not chunk_ids:
            raise ValueError(f"Case {index + 1} requires evidence_chunk_ids.")
        if known and any(chunk_id not in known for chunk_id in chunk_ids):
            raise ValueError(f"Case {index + 1} references an unknown chunk.")
        if case.get("split") not in {"calibration", "heldout"}:
            raise ValueError(f"Case {index + 1} requires calibration or heldout split.")


def _predicted_label(verdict: str) -> str:
    if verdict == "supported":
        return "supported"
    if verdict in {"contradicted", "disputed"}:
        return "contradiction"
    return "neutral"


def grounding_benchmark_fingerprint(cases: Sequence[Dict[str, Any]], corpus_items: Sequence[Dict[str, Any]]) -> str:
    by_id = {item["chunk_id"]: item for item in corpus_items}
    payload = []
    for case in sorted(cases, key=lambda value: str(value.get("case_id", ""))):
        payload.append({
            "case": case,
            "evidence": [
                {
                    "chunk_id": chunk_id,
                    "generation_text": " ".join(str(by_id[chunk_id].get("generation_text", by_id[chunk_id].get("text", ""))).split()),
                }
                for chunk_id in case["evidence_chunk_ids"]
            ],
        })
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _score_case_pairs(
    cases: Sequence[Dict[str, Any]], by_id: Dict[str, Dict[str, Any]], verifier, policy: GroundingPolicy
) -> Tuple[Dict[Tuple[str, str], Dict[str, float]], float]:
    pairs = []
    for case in cases:
        for chunk_id in case["evidence_chunk_ids"]:
            pairs.append((select_evidence_premise(by_id[chunk_id], case["claim"], policy), case["claim"]))
    started = time.perf_counter()
    scores = verifier.score(pairs)
    latency = (time.perf_counter() - started) * 1000
    if len(scores) != len(pairs):
        raise ValueError("Verifier returned a different number of scores than benchmark pairs.")
    return {pair: score for pair, score in zip(pairs, scores)}, latency


def _report_from_scores(
    cases: List[Dict[str, Any]], by_id: Dict[str, Dict[str, Any]], verifier,
    policy: GroundingPolicy, score_map: Dict[Tuple[str, str], Dict[str, float]], batch_latency_ms: float,
) -> Dict[str, Any]:
    class CachedVerifier:
        model_name = verifier.model_name
        model_revision = verifier.model_revision

        @staticmethod
        def score(pairs):
            return [score_map[pair] for pair in pairs]

    rows = []
    for case in cases:
        evidence_items = [by_id[chunk_id] for chunk_id in case["evidence_chunk_ids"]]
        legacy = [(1.0 - index * 0.001, item) for index, item in enumerate(evidence_items)]
        draft = StructuredDraft(answerable=True, claims=[DraftClaim(
            text=case["claim"], cited_chunk_ids=case["evidence_chunk_ids"][:3], provenance="generated"
        )])
        run = verify_claims(draft, legacy, verifier=CachedVerifier(), policy=policy)
        claim = (run.result.accepted_claims + run.result.rejected_claims)[0]
        predicted = _predicted_label(claim.verdict)
        rows.append({
            "case_id": case.get("case_id", ""), "split": case["split"],
            "category": case.get("category", "general"), "claim": case["claim"],
            "expected": case["label"], "predicted": predicted,
            "verdict": claim.verdict, "correct": predicted == case["label"],
            "entailment_score": claim.entailment_score,
            "contradiction_score": claim.contradiction_score,
            "latency_ms": batch_latency_ms / max(1, len(cases)),
        })
    report = summarize_grounding_rows(rows)
    report["batch_latency_ms"] = batch_latency_ms
    report["verifier"] = {"model": verifier.model_name, "revision": verifier.model_revision}
    report["policy"] = policy.to_dict()
    return report


def run_grounding_benchmark(
    cases: List[Dict[str, Any]], corpus_items: List[Dict[str, Any]], verifier,
    policy: GroundingPolicy | None = None,
) -> Dict[str, Any]:
    by_id = {item["chunk_id"]: item for item in corpus_items}
    validate_grounding_benchmark(cases, by_id)
    active_policy = policy or default_grounding_policy()
    score_map, latency = _score_case_pairs(cases, by_id, verifier, active_policy)
    report = _report_from_scores(cases, by_id, verifier, active_policy, score_map, latency)
    report["benchmark_fingerprint"] = grounding_benchmark_fingerprint(cases, corpus_items)
    return report


def _stratified_folds(cases: Sequence[Dict[str, Any]], k: int) -> List[List[Dict[str, Any]]]:
    """Split cases into k folds, balanced by label via round-robin within each label group.

    Deterministic (sorted by case_id before assignment) so the same cases always land in the
    same fold across runs -- calibration selection must be reproducible.
    """
    folds: List[List[Dict[str, Any]]] = [[] for _ in range(k)]
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_label[case["label"]].append(case)
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda case: str(case.get("case_id", "")))
        for index, case in enumerate(ordered):
            folds[index % k].append(case)
    return folds


def calibrate_grounding_policy(
    cases: List[Dict[str, Any]], corpus_items: List[Dict[str, Any]], verifier,
    base_policy: GroundingPolicy | None = None, cv_folds: int = 4,
) -> Dict[str, Any]:
    if not cases or any(case.get("split") != "calibration" for case in cases):
        raise ValueError("Calibration accepts calibration cases only; held-out cases are sealed.")
    by_id = {item["chunk_id"]: item for item in corpus_items}
    validate_grounding_benchmark(cases, by_id)
    base = base_policy or default_grounding_policy()
    # Folds are fixed by the case set alone (label-stratified, not policy-dependent), so they're
    # computed once and reused for every candidate's cross-validated score below. This is nested
    # cross-validation *within* the 48 calibration cases only -- held-out is never touched, per
    # docs/QUALITY_GATE_RELEASE.md's "the 48 grounding calibration cases are the only cases used
    # to select premise strategy and thresholds."
    folds = _stratified_folds(cases, cv_folds)
    candidates = []
    for strategy in ("atomic_sentence", "context_envelope"):
        premise_version = "atomic-sentence-v1" if strategy == "atomic_sentence" else "context-envelope-v1"
        scoring_policy = replace(base, premise_strategy=strategy, premise_version=premise_version)
        score_map, latency = _score_case_pairs(cases, by_id, verifier, scoring_policy)
        for entailment in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            for contradiction in (0.65, 0.70, 0.75, 0.80, 0.85):
                for relevance in (0.20, 0.30, 0.40):
                    policy = replace(
                        scoring_policy, model_name=verifier.model_name, model_revision=verifier.model_revision,
                        entailment_threshold=entailment, contradiction_threshold=contradiction,
                        conflict_relevance_threshold=relevance,
                    )
                    # Pooled metrics over all 48 cases decide eligibility, unchanged --
                    # QUALITY_GATE_RELEASE.md's safety bar (precision>=0.95,
                    # contradiction_recall>=0.85) is a fixed constraint, not something this
                    # methodology change alters.
                    report = _report_from_scores(cases, by_id, verifier, policy, score_map, latency)
                    precision = report["per_label"]["supported"]["precision"]
                    contradiction_recall = report["per_label"]["contradiction"]["recall"]
                    eligible = precision >= 0.95 and contradiction_recall >= 0.85
                    # Cross-validated macro_f1: many candidates tie exactly on the pooled score
                    # (48 cases is too little data to discriminate ~20+ threshold combinations
                    # that classify them identically), so pooled macro_f1 alone has no signal
                    # left to select on. Reusing the same cached score_map, score each candidate
                    # per fold instead -- a candidate whose macro_f1 stays high and stable across
                    # folds generalizes to unseen cases more reliably than one that reaches the
                    # same pooled average by acing most folds and stumbling on one.
                    fold_macro_f1s = [
                        _report_from_scores(fold, by_id, verifier, policy, score_map, latency)["macro_f1"]
                        for fold in folds if fold
                    ]
                    mean_fold_macro_f1 = statistics.fmean(fold_macro_f1s) if fold_macro_f1s else 0.0
                    fold_macro_f1_stdev = statistics.pstdev(fold_macro_f1s) if len(fold_macro_f1s) > 1 else 0.0
                    cv_robust_macro_f1 = mean_fold_macro_f1 - fold_macro_f1_stdev
                    candidates.append({
                        "policy": policy.to_dict(), "eligible": eligible,
                        "macro_f1": report["macro_f1"], "supported_precision": precision,
                        "contradiction_recall": contradiction_recall,
                        "latency_ms": report["latency_ms"]["p95"],
                        "cv_fold_macro_f1_mean": mean_fold_macro_f1,
                        "cv_fold_macro_f1_stdev": fold_macro_f1_stdev,
                        "cv_robust_macro_f1": cv_robust_macro_f1,
                    })
    ranked = sorted(
        candidates,
        key=lambda row: (
            not row["eligible"], -row["cv_robust_macro_f1"], -row["macro_f1"],
            row["latency_ms"], row["policy"]["policy_id"],
        ),
    )
    selected = next((row for row in ranked if row["eligible"]), None)
    return {
        "schema_version": GROUNDING_ARTIFACT_SCHEMA_VERSION,
        "calibration_only": True,
        "calibration_fingerprint": grounding_benchmark_fingerprint(cases, corpus_items),
        "cv_folds": cv_folds,
        "selected": selected,
        "top_candidates": ranked[:20],
        "candidate_count": len(candidates),
    }


def summarize_grounding_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    confusion = {expected: {predicted: 0 for predicted in LABELS} for expected in LABELS}
    for row in rows:
        confusion[row["expected"]][row["predicted"]] += 1
    per_label = {}
    f1_values = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[expected][label] for expected in LABELS if expected != label)
        fn = sum(confusion[label][predicted] for predicted in LABELS if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[label].values())}
        f1_values.append(f1)
    categories = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(float(row["correct"]))
    heldout = [row for row in rows if row["split"] == "heldout"]
    heldout_report = summarize_grounding_rows(heldout) if heldout and len(heldout) != len(rows) else None
    report = {
        "schema_version": "1.0", "n": len(rows),
        "accuracy": sum(float(row["correct"]) for row in rows) / max(1, len(rows)),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_label": per_label, "confusion_matrix": confusion,
        "category_accuracy": {key: sum(values) / len(values) for key, values in sorted(categories.items())},
        "latency_ms": {
            "p50": percentile([row["latency_ms"] for row in rows], 0.50),
            "p95": percentile([row["latency_ms"] for row in rows], 0.95),
        },
        "rows": rows,
        "policy_comparison": {
            "structured_unfiltered": {
                "displayed_claims": len(rows),
                "unsupported_exposed": sum(row["expected"] != "supported" for row in rows),
            },
            "strict_grounded": {
                "displayed_claims": sum(row["predicted"] == "supported" for row in rows),
                "unsupported_exposed": sum(
                    row["predicted"] == "supported" and row["expected"] != "supported" for row in rows
                ),
            },
        },
        "largest_safety_improvements": sorted(
            [row for row in rows if row["expected"] != "supported" and row["predicted"] != "supported"],
            key=lambda row: (-row["contradiction_score"], row["case_id"]),
        )[:10],
        "largest_answer_regressions": sorted(
            [row for row in rows if row["expected"] == "supported" and row["predicted"] != "supported"],
            key=lambda row: (row["entailment_score"], row["case_id"]),
        )[:10],
    }
    if heldout_report:
        heldout_report.pop("heldout", None)
        report["heldout"] = heldout_report
    return report


def grounding_promotion_gate(
    claim_report: Dict[str, Any], qa_report: Dict[str, Any], baseline_qa_report: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    measured = claim_report.get("heldout") or claim_report
    support_precision = measured["per_label"]["supported"]["precision"]
    contradiction_recall = measured["per_label"]["contradiction"]["recall"]
    displayed_support = qa_report.get("grounding", {}).get("displayed_claim_support")
    citation_coverage = qa_report.get("grounding", {}).get("claim_citation_coverage")
    answer_coverage = qa_report.get("grounding", {}).get("answer_coverage") or 0.0
    verification_p95 = qa_report.get("latency_ms", {}).get("verification_p95") or 0.0
    baseline_qa_report = baseline_qa_report or {}
    accuracy_delta = qa_report.get("accuracy", 0.0) - baseline_qa_report.get("accuracy", 0.0)
    abstention_delta = qa_report.get("abstention_accuracy", 0.0) - baseline_qa_report.get("abstention_accuracy", 0.0)
    unsupported_exposed = measured.get("policy_comparison", {}).get("strict_grounded", {}).get("unsupported_exposed")
    matching_retrieval = bool(
        baseline_qa_report
        and qa_report.get("retrieval_results_fingerprint")
        == baseline_qa_report.get("retrieval_results_fingerprint")
    )
    matching_drafts = bool(
        baseline_qa_report
        and qa_report.get("draft_claims_fingerprint")
        == baseline_qa_report.get("draft_claims_fingerprint")
    )
    matching_policy = bool(
        baseline_qa_report
        and qa_report.get("grounding_policy_id")
        == baseline_qa_report.get("grounding_policy_id")
    )
    gates = {
        "matching_retrieval": matching_retrieval,
        "matching_draft_claims": matching_drafts,
        "matching_policy": matching_policy,
        "supported_precision": support_precision >= 0.95,
        "macro_f1": measured["macro_f1"] >= 0.85,
        "contradiction_recall": contradiction_recall >= 0.85,
        "displayed_claim_support": displayed_support == 1.0,
        "citation_coverage": citation_coverage == 1.0,
        "citation_validity": qa_report.get("citation_validity") == 1.0,
        "zero_unsupported_exposure": unsupported_exposed == 0,
        "answer_coverage": answer_coverage >= 0.85,
        "answer_accuracy_regression": bool(baseline_qa_report) and accuracy_delta >= -0.02,
        "abstention_regression": bool(baseline_qa_report) and abstention_delta >= -0.01,
        "verification_latency": verification_p95 < 1500.0,
    }
    return {"passed": all(gates.values()), "gates": gates, "measured": {
        "supported_precision": support_precision, "macro_f1": measured["macro_f1"],
        "contradiction_recall": contradiction_recall, "answer_coverage": answer_coverage,
        "unsupported_exposed": unsupported_exposed,
        "answer_accuracy_delta": accuracy_delta, "abstention_accuracy_delta": abstention_delta,
        "verification_p95_ms": verification_p95,
    }}


def compare_grounding_policies(qa_report: Dict[str, Any]) -> Dict[str, Any]:
    """Compare unfiltered and strict rendering from the same cached draft claims."""
    rows = []
    for result in qa_report.get("results", []):
        grounding = result.get("grounding", {})
        accepted = grounding.get("accepted_claims", [])
        rejected = grounding.get("rejected_claims", [])
        rows.append({
            "question": result.get("question", ""), "category": result.get("category", "general"),
            "draft_claims": len(accepted) + len(rejected), "strict_claims": len(accepted),
            "unsafe_claims_removed": len(rejected),
            "removed_verdicts": [claim.get("verdict", "") for claim in rejected],
        })
    n = max(1, len(rows))
    comparison = {
        "same_cached_drafts": bool(qa_report.get("draft_claims_fingerprint", True)),
        "structured_unfiltered": {
            "displayed_claims": sum(row["draft_claims"] for row in rows),
            "unsafe_claims_exposed": sum(row["unsafe_claims_removed"] for row in rows),
            "answer_coverage": sum(row["draft_claims"] > 0 for row in rows) / n,
        },
        "strict_grounded": {
            "displayed_claims": sum(row["strict_claims"] for row in rows),
            "unsafe_claims_exposed": 0,
            "answer_coverage": sum(row["strict_claims"] > 0 for row in rows) / n,
        },
        "largest_safety_improvements": sorted(
            [row for row in rows if row["unsafe_claims_removed"]],
            key=lambda row: (-row["unsafe_claims_removed"], row["question"]),
        )[:10],
        "largest_answer_regressions": sorted(
            [row for row in rows if row["draft_claims"] and not row["strict_claims"]],
            key=lambda row: row["question"],
        )[:10],
    }
    return comparison


def save_grounding_artifact(
    report: Dict[str, Any], gate: Dict[str, Any], output_root: str,
    qa_policy_comparison: Dict[str, Any] | None = None,
) -> str:
    directory = os.path.join(output_root, "grounding")
    ensure_dir(directory)
    path = os.path.join(directory, "grounding_comparison.json")
    write_json(path, {
        "schema_version": GROUNDING_ARTIFACT_SCHEMA_VERSION, "report": report, "promotion_gate": gate,
        "qa_policy_comparison": qa_policy_comparison or {},
    })
    return path
