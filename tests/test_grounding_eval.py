import json
from pathlib import Path

import pytest

from backend.contextual_chunking import build_contextual_chunks
from backend.document_parsers import parse_document
from backend.grounding_eval import (
    calibrate_grounding_policy,
    compare_grounding_policies,
    grounding_benchmark_fingerprint,
    grounding_promotion_gate,
    run_grounding_benchmark,
    validate_grounding_benchmark,
)
from backend.grounding_policy import GroundingPolicy


class LabelVerifier:
    model_name = "label-verifier"
    model_revision = "test"

    def score(self, pairs):
        rows = []
        for premise, claim in pairs:
            if "contradiction" in claim:
                rows.append({"contradiction": .95, "entailment": .02, "neutral": .03})
            elif "neutral" in claim:
                rows.append({"contradiction": .02, "entailment": .03, "neutral": .95})
            else:
                rows.append({"contradiction": .02, "entailment": .95, "neutral": .03})
        return rows


def test_claim_benchmark_metrics_and_promotion_gate():
    corpus = [{"chunk_id": "c1", "text": "Evidence.", "generation_text": "Evidence.", "source": "x", "page": 1}]
    cases = [
        {"claim": "supported", "evidence_chunk_ids": ["c1"], "label": "supported", "split": "calibration", "category": "a"},
        {"claim": "neutral", "evidence_chunk_ids": ["c1"], "label": "neutral", "split": "heldout", "category": "b"},
        {"claim": "contradiction", "evidence_chunk_ids": ["c1"], "label": "contradiction", "split": "heldout", "category": "c"},
    ]
    report = run_grounding_benchmark(cases, corpus, LabelVerifier())
    assert report["accuracy"] == 1.0
    assert report["heldout"]["macro_f1"] < 1.0  # no supported held-out example
    qa = {"grounding": {"displayed_claim_support": 1.0, "claim_citation_coverage": 1.0, "answer_coverage": .9}, "latency_ms": {"verification_p95": 100}}
    assert not grounding_promotion_gate(report, qa)["passed"]


def test_public_grounding_benchmark_has_96_valid_stable_cases():
    root = Path(__file__).resolve().parents[1]
    chunks = []
    for path in sorted((root / "data" / "public_demo").glob("*.md")):
        parsed = parse_document(str(path), source_name=path.name)
        chunks.extend(build_contextual_chunks(parsed.document, parsed.version, parsed.blocks))
    cases = json.loads((root / "data" / "grounding_benchmark.json").read_text(encoding="utf-8"))
    validate_grounding_benchmark(cases, [chunk.chunk_id for chunk in chunks])
    assert len(cases) == 96
    assert sum(case["split"] == "calibration" for case in cases) == 48
    assert sum(case["split"] == "heldout" for case in cases) == 48
    assert len({case["category"] for case in cases}) == 6


def test_grounding_benchmark_rejects_bad_labels_and_unknown_chunks():
    with pytest.raises(ValueError):
        validate_grounding_benchmark([{"claim": "x", "label": "bad", "evidence_chunk_ids": ["c1"], "split": "heldout"}], ["c1"])
    with pytest.raises(ValueError):
        validate_grounding_benchmark([{"claim": "x", "label": "supported", "evidence_chunk_ids": ["missing"], "split": "heldout"}], ["c1"])


def test_calibration_rejects_heldout_and_fingerprints_evidence():
    corpus = [{"chunk_id": "c1", "text": "Evidence.", "generation_text": "Evidence."}]
    heldout = [{"case_id": "h", "claim": "supported", "evidence_chunk_ids": ["c1"], "label": "supported", "split": "heldout"}]
    with pytest.raises(ValueError, match="sealed"):
        calibrate_grounding_policy(heldout, corpus, LabelVerifier(), GroundingPolicy())
    calibration = [
        {"case_id": "s", "claim": "supported", "evidence_chunk_ids": ["c1"], "label": "supported", "split": "calibration"},
        {"case_id": "n", "claim": "neutral", "evidence_chunk_ids": ["c1"], "label": "neutral", "split": "calibration"},
        {"case_id": "c", "claim": "contradiction", "evidence_chunk_ids": ["c1"], "label": "contradiction", "split": "calibration"},
    ]
    result = calibrate_grounding_policy(calibration, corpus, LabelVerifier(), GroundingPolicy())
    assert result["calibration_only"] and result["selected"]
    changed = [{"chunk_id": "c1", "text": "Changed.", "generation_text": "Changed."}]
    assert grounding_benchmark_fingerprint(calibration, corpus) != grounding_benchmark_fingerprint(calibration, changed)


def test_policy_comparison_reuses_drafts_and_counts_removed_claims():
    report = {"results": [{
        "question": "q", "category": "test", "grounding": {
            "accepted_claims": [{"verdict": "supported"}],
            "rejected_claims": [{"verdict": "unsupported"}],
        },
    }]}
    comparison = compare_grounding_policies(report)
    assert comparison["same_cached_drafts"]
    assert comparison["structured_unfiltered"]["unsafe_claims_exposed"] == 1
    assert comparison["strict_grounded"]["unsafe_claims_exposed"] == 0
