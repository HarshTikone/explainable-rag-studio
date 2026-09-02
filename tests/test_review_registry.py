from concurrent.futures import ThreadPoolExecutor

from backend.review_registry import ReviewRegistry


PAYLOAD = {
    "text": "A disputed claim.", "verdict": "disputed",
    "evidence": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
}


def test_review_queue_deduplicates_and_keeps_append_only_decisions(tmp_path):
    registry = ReviewRegistry(str(tmp_path / "reviews.db"))
    first = registry.enqueue(PAYLOAD, "disputed", "config")
    second = registry.enqueue(PAYLOAD, "low_confidence", "config")
    assert first == second
    assert len(registry.list_cases()) == 1
    assert registry.decide(first, "contradicted", "reviewer", "Evidence is explicit")
    assert registry.get_case(first)["status"] == "resolved"
    assert registry.get_case(first)["decisions"][0]["decision"] == "contradicted"
    assert "Evidence is explicit" in registry.export_jsonl("all")


def test_review_queue_handles_concurrent_duplicate_writes(tmp_path):
    registry = ReviewRegistry(str(tmp_path / "reviews.db"))
    with ThreadPoolExecutor(max_workers=6) as pool:
        case_ids = list(pool.map(lambda _: registry.enqueue(PAYLOAD, "disputed", "config"), range(12)))
    assert len(set(case_ids)) == 1
    assert len(registry.list_cases()) == 1


def test_review_decision_validation_and_missing_case(tmp_path):
    registry = ReviewRegistry(str(tmp_path / "reviews.db"))
    assert not registry.decide("missing", "supported")
    case_id = registry.enqueue(PAYLOAD, "disputed", "config")
    try:
        registry.decide(case_id, "maybe")
        assert False, "Expected invalid review decision to fail"
    except ValueError:
        pass

