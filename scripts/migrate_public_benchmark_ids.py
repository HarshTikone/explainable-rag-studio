"""Relabel the public benchmark with stable IDs and emit legacy aliases."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.contextual_chunking import build_contextual_chunks
from backend.document_parsers import parse_document
from backend.experiments import corpus_fingerprint


CORPUS_DIR = ROOT / "data" / "public_demo"
BENCHMARK_PATH = ROOT / "data" / "public_demo_benchmark.json"
ALIAS_PATH = ROOT / "data" / "public_demo_chunk_aliases.json"


def build_mapping():
    chunks = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        parsed = parse_document(str(path), source_name=path.name)
        chunks.extend(build_contextual_chunks(parsed.document, parsed.version, parsed.blocks))
    old_ids = [f"c{index:06d}" for index in range(1, len(chunks) + 1)]
    mapping = dict(zip(old_ids, [chunk.chunk_id for chunk in chunks]))
    return chunks, mapping


def main():
    chunks, mapping = build_mapping()
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    for item in benchmark:
        item["relevant_chunk_ids"] = [mapping.get(chunk_id, chunk_id) for chunk_id in item.get("relevant_chunk_ids", [])]
    BENCHMARK_PATH.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    aliases = {
        "schema_version": "1.0",
        "old_to_new": mapping,
        "new_to_old": {new: old for old, new in mapping.items()},
        "context_aware_corpus_fingerprint": corpus_fingerprint([chunk.to_dict() for chunk in chunks]),
    }
    ALIAS_PATH.write_text(json.dumps(aliases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Mapped {len(mapping)} legacy IDs and relabeled {len(benchmark)} questions.")


if __name__ == "__main__":
    main()
