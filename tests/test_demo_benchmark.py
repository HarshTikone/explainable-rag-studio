import json
from pathlib import Path

from backend.contextual_chunking import build_contextual_chunks
from backend.document_parsers import parse_document


def test_public_demo_benchmark_is_complete_and_sanitized():
    items = json.loads(Path("data/public_demo_benchmark.json").read_text(encoding="utf-8"))
    assert len(items) == 60
    assert {item["category"] for item in items} == {"exact_term", "identifier", "paraphrase", "hard_negative", "multi_hop", "unanswerable"}
    assert all(sum(1 for item in items if item["category"] == category) == 10 for category in {item["category"] for item in items})
    assert all(item["relevant_chunk_ids"] == [] for item in items if not item["answerable"])
    assert all(all(chunk_id.startswith("chk_") for chunk_id in item["relevant_chunk_ids"]) for item in items)
    assert all("expected_source_versions" in item for item in items)


def test_public_demo_documents_have_deterministic_order():
    names = [path.name for path in sorted(Path("data/public_demo").glob("*.md"))]
    assert len(names) == 60
    assert names[0] == "01_platform.md"
    assert names[-1] == "60_lifecycle_incident.md"


def test_benchmark_labels_resolve_against_rebuilt_default_chunks():
    chunks = []
    for path in sorted(Path("data/public_demo").glob("*.md")):
        parsed = parse_document(str(path), source_name=path.name)
        chunks.extend(build_contextual_chunks(parsed.document, parsed.version, parsed.blocks))
    assert len(chunks) == 60
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    benchmark = json.loads(Path("data/public_demo_benchmark.json").read_text(encoding="utf-8"))
    labeled_ids = {chunk_id for item in benchmark for chunk_id in item["relevant_chunk_ids"]}
    assert labeled_ids <= chunk_ids
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    for item in benchmark:
        assert set(item["expected_source_versions"]) == {by_id[chunk_id].source_version for chunk_id in item["relevant_chunk_ids"]}
    aliases = json.loads(Path("data/public_demo_chunk_aliases.json").read_text(encoding="utf-8"))
    assert len(aliases["old_to_new"]) == 60
    assert set(aliases["old_to_new"].values()) == chunk_ids
    assert aliases["new_to_old"] == {new: old for old, new in aliases["old_to_new"].items()}
