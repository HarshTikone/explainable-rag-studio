from typing import List, Dict, Any, Tuple

def pick_top_citations(retrieved: List[Tuple[float, Dict[str, Any]]], max_cites: int = 3):
    """
    Picks up to max_cites chunks for citations (best scores).
    """
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
