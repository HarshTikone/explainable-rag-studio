"""Build the bundled public demo corpus into a deployable FAISS index."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import SETTINGS
from backend.contextual_chunking import build_contextual_chunks
from backend.document_parsers import parse_document
from backend.vectorstore import FaissStore


def load_public_corpus(corpus_dir: str):
    chunks = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        parsed = parse_document(str(path), source_name=path.name)
        chunks.extend(build_contextual_chunks(parsed.document, parsed.version, parsed.blocks))
    items = [chunk.to_dict() for chunk in chunks]
    if len(items) != 60:
        raise ValueError(f"Public corpus must produce exactly 60 deterministic chunks; got {len(items)}.")
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/public_demo")
    parser.add_argument("--output", default=SETTINGS.index_dir)
    args = parser.parse_args()

    items = load_public_corpus(args.corpus)
    if SETTINGS.low_memory_demo:
        store = FaissStore(args.output)
        store.build(np.ones((len(items), 1), dtype="float32"), items)
    else:
        from backend.embeddings import Embedder
        from backend.release_quality import build_release_store
        store = build_release_store(args.output, items, Embedder(SETTINGS.embedding_model))
    print(f"Built public demo index: {store.index.ntotal} chunks in {args.output}")


if __name__ == "__main__":
    main()
