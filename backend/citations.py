from typing import List, Dict, Any, Tuple
from .retriever import RetrievalResult

def pick_top_citations(retrieved, max_cites: int = 3):
    """
    Picks up to max_cites chunks for citations (best scores).
    """
    if isinstance(retrieved, RetrievalResult):
        sorted_items = retrieved.as_legacy()
    else:
        sorted_items = sorted(retrieved, key=lambda x: x[0], reverse=True)
    cite = []
    for score, item in sorted_items[:max_cites]:
        cite.append({
            "chunk_id": item["chunk_id"],
            "source": item["source"],
            "page": item["page"],
            "score": score
        })
    return cite
