from typing import List, Dict, Any, Tuple
import os
import numpy as np
import faiss

from .utils import ensure_dir, write_json, read_json

INDEX_FILE = "faiss.index"
META_FILE = "meta.json"

class FaissStore:
    """
    FAISS index + metadata store.
    Stores:
      - FAISS vectors in index/faiss.index
      - Metadata (aligned by vector position) in index/meta.json
    """
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        ensure_dir(index_dir)
        self.index_path = os.path.join(index_dir, INDEX_FILE)
        self.meta_path = os.path.join(index_dir, META_FILE)

        self.index = None
        self.meta = {"items": []}

    def build(self, vectors: np.ndarray, items: List[Dict[str, Any]]) -> None:
        """
        vectors: (N, d) float32 normalized
        items: list of chunk metadata + text
        """
        d = vectors.shape[1]
        index = faiss.IndexFlatIP(d)  # inner product, good for normalized embeddings (cosine)
        index.add(vectors)

        self.index = index
        self.meta = {"items": items}
        self.save()

    def save(self) -> None:
        if self.index is None:
            return
        faiss.write_index(self.index, self.index_path)
        write_json(self.meta_path, self.meta)

    def load(self) -> bool:
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            return False
        self.index = faiss.read_index(self.index_path)
        self.meta = read_json(self.meta_path)
        return True

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Returns list of (score, item) sorted best-first.
        """
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Build or load first.")

        scores, idxs = self.index.search(query_vec, top_k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        results = []
        for s, i in zip(scores, idxs):
            if i == -1:
                continue
            item = self.meta["items"][i]
            results.append((float(s), item))
        return results
