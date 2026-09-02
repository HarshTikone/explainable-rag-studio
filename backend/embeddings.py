from typing import List
import numpy as np

class Embedder:
    """
    Local embedding model wrapper using SentenceTransformers.
    """
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")
