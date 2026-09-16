"""Build the bundled public demo corpus into a deployable FAISS index."""
from __future__ import annotations

import argparse

from backend.config import SETTINGS
from backend.embeddings import Embedder
from backend.release_quality import build_release_store, load_public_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/public_demo")
    parser.add_argument("--output", default=SETTINGS.index_dir)
    args = parser.parse_args()

    items = load_public_corpus(args.corpus)
    store = build_release_store(args.output, items, Embedder(SETTINGS.embedding_model))
    print(f"Built public demo index: {store.index.ntotal} chunks in {args.output}")


if __name__ == "__main__":
    main()
