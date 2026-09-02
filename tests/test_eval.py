import json

import pytest

from backend.eval import (
    citation_metrics, is_abstention, load_eval_set, percentile,
    retrieval_metrics, run_eval, simple_accuracy, validate_eval_set,
)


def test_retrieval_metrics_respect_rank_and_labels():
    metrics = retrieval_metrics(["c3", "c1", "c2"], ["c1", "c9"])
    assert metrics["hit_rate"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 1 / 3
    assert metrics["reciprocal_rank"] == 0.5
    assert 0 < metrics["ndcg"] < 1


def test_unlabeled_retrieval_metrics_are_not_fake_zeroes():
    assert all(value is None for value in retrieval_metrics(["c1"], []).values())


def test_citations_must_reference_retrieved_chunks():
    metrics = citation_metrics([{"chunk_id": "c1"}, {"chunk_id": "c9"}], ["c1", "c2"])
    assert metrics["citation_validity"] == 0.5
    assert citation_metrics([], [], allow_empty=True)["citation_validity"] == 1.0


def test_eval_uses_claim_and_answerable_denominators(tmp_path):
    items = [
        {"question": "answerable", "answerable": True},
        {"question": "unanswerable", "answerable": False},
    ]

    def ask(question):
        if question == "unanswerable":
            return {
                "answer": "I don't know based on the retrieved evidence.", "citations": [], "retrieved": [],
                "grounding": {"accepted_claims": [], "rejected_claims": [], "latency_ms": {}},
            }
        claim = {"verdict": "supported", "cited_chunk_ids": ["c1"]}
        return {
            "answer": "Supported [c1]", "citations": [{"chunk_id": "c1"}], "retrieved": [{"chunk_id": "c1"}],
            "grounding": {"accepted_claims": [claim], "rejected_claims": [], "latency_ms": {}},
        }

    report = run_eval(items, ask, str(tmp_path), write_report=False)
    assert report["grounding"]["answer_coverage"] == 1.0
    assert report["grounding"]["displayed_claim_support"] == 1.0
    assert report["grounding"]["claim_citation_coverage"] == 1.0
    assert report["citation_validity"] == 1.0


def test_abstention_and_baseline_accuracy():
    assert is_abstention("I don't know based on the supplied context.")
    assert simple_accuracy("FAISS is used for indexing.", "faiss") == 1.0


def test_ndcg_is_one_for_perfect_ranking_and_ignores_duplicates():
    assert retrieval_metrics(["c1", "c2"], ["c1", "c2"])["ndcg"] == 1.0
    assert retrieval_metrics(["c1", "c1", "c2"], ["c1", "c2"])["ndcg"] == 1.0
    reversed_score = retrieval_metrics(["x", "c1"], ["c1"])["ndcg"]
    assert 0 < reversed_score < 1


def test_eval_loading_validation_and_percentiles(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([{"question": "Q?", "expected": "A"}]), encoding="utf-8")
    assert load_eval_set(str(path))[0]["question"] == "Q?"
    assert percentile([], .95) is None
    assert percentile([10], .95) == 10
    with pytest.raises(ValueError):
        validate_eval_set([])
    with pytest.raises(ValueError):
        validate_eval_set([{"question": ""}])
    with pytest.raises(ValueError):
        validate_eval_set([{"question": "Q", "relevant_chunk_ids": "c1"}])
