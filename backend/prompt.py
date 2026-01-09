from typing import List, Dict, Any

def build_context(retrieved: List[Dict[str, Any]]) -> str:
    """
    Creates a structured context block with chunk ids and metadata.
    """
    parts = []
    for r in retrieved:
        parts.append(
            f"[{r['chunk_id']}] Source: {r['source']} | Page: {r['page']}\n"
            f"{r['text']}\n"
        )
    return "\n---\n".join(parts)

def system_prompt() -> str:
    return (
        "You are a helpful assistant that answers questions using ONLY the provided context.\n"
        "Rules:\n"
        "1) If the answer is not in the context, say you don't know.\n"
        "2) Provide a concise answer.\n"
        "3) Always include 2–3 citations referencing chunk ids like [c000123].\n"
        "4) Do not invent sources.\n"
    )

def user_prompt(question: str, context: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Return format:\n"
        "Answer: <your answer>\n"
        "Citations: [chunk_id], [chunk_id], [chunk_id]\n"
    )
