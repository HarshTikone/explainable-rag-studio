from typing import List, Dict, Any, Tuple
import numpy as np

from .vectorstore import FaissStore

def mmr_select(
    query_vec: np.ndarray,
    candidates: List[Tuple[float, Dict[str, Any]]],
    candidate_vecs: np.ndarray,
    k: int,
    lambda_mult: float = 0.5
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Simple MMR: balances relevance (to query) and diversity (between selected chunks).
    candidates: list of (score, item) for N
    candidate_vecs: (N, d) normalized vectors aligned with candidates
    """
    if not candidates:
        return []

    selected = []
    selected_idxs = []

    # relevance is the score already from FAISS (cosine via IP)
    rel = np.array([c[0] for c in candidates], dtype="float32")

    # similarity between candidates
    sim = candidate_vecs @ candidate_vecs.T  # (N, N)

    # start with best relevance
    first = int(rel.argmax())
    selected_idxs.append(first)
    selected.append(candidates[first])

    while len(selected) < min(k, len(candidates)):
        remaining = [i for i in range(len(candidates)) if i not in selected_idxs]
        best_i = None
        best_score = -1e9
        for i in remaining:
            diversity = max(sim[i, j] for j in selected_idxs)
            mmr_score = lambda_mult * rel[i] - (1 - lambda_mult) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_i = i
        selected_idxs.append(best_i)
        selected.append(candidates[best_i])

    return selected

def retrieve(
    store: FaissStore,
    embed_query_fn,
    query: str,
    top_k: int,
    use_mmr: bool = True
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Retrieves chunks. If MMR enabled, expands candidates and selects diverse top_k.
    """
    qv = embed_query_fn(query)  # (1, d)
    if not use_mmr:
        return store.search(qv, top_k)

    # Get more candidates first, then select top_k via MMR
    candidate_k = min(30, max(top_k * 5, top_k))
    cands = store.search(qv, candidate_k)

    # Need candidate vectors for MMR diversity:
    # We approximate by re-embedding candidate texts (fast enough for demo),
    # or you can persist vectors in meta for production.
    texts = [c[1]["text"] for c in cands]
    vecs = embed_query_fn.__self__.embed_texts(texts)  # using Embedder instance
    selected = mmr_select(qv, cands, vecs, k=top_k)

    return selected
