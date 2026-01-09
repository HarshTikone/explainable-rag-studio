from typing import Dict, Any, List, Tuple
import re

from .prompt import build_context
from .citations import pick_top_citations

def _system_rules() -> str:
    return (
        "You are a helpful assistant that answers questions using ONLY the provided context.\n"
        "Rules:\n"
        "1) If the answer is not in the context, say you don't know.\n"
        "2) Provide a concise answer.\n"
        "3) Always include 2–3 citations referencing chunk ids like [c000123].\n"
        "4) Do not invent sources.\n"
    )

def extractive_answer(question: str, retrieved_items: List[Tuple[float, Dict[str, Any]]]) -> str:
    """
    Non-LLM fallback: returns the most relevant chunk excerpt.
    Keeps the app usable even without an API key.
    """
    if not retrieved_items:
        return "I don't know."

    top = sorted(retrieved_items, key=lambda x: x[0], reverse=True)[0][1]["text"]
    sentences = re.split(r"(?<=[.!?])\s+", top)
    return " ".join(sentences[:3]).strip() or "I don't know."

def answer_with_optional_llm(
    question: str,
    retrieved_items: List[Tuple[float, Dict[str, Any]]],
    use_gemini: bool,
    gemini_client=None,
    gemini_model: str = ""
) -> Dict[str, Any]:
    """
    Returns:
      {
        answer: str,
        citations: [{chunk_id, source, page, score}],
        context: str
      }
    """
    citations = pick_top_citations(retrieved_items, max_cites=3)
    retrieved = [it for _, it in retrieved_items]
    context = build_context(retrieved)

    if not retrieved_items:
        return {"answer": "I don't know.", "citations": [], "context": context}

    if not use_gemini:
        ans = extractive_answer(question, retrieved_items)
        return {"answer": ans, "citations": citations, "context": context}

    # Gemini prompt: keep it explicit and grounded
    prompt = (
        f"{_system_rules()}\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Return ONLY the final answer text. Include citations inline like [c000123]."
    )

    resp = gemini_client.models.generate_content(
        model=gemini_model,
        contents=prompt
    )

    # google-genai exposes response text via resp.text
    answer = (resp.text or "").strip() if hasattr(resp, "text") else str(resp).strip()
    if not answer:
        answer = "I don't know."

    return {"answer": answer, "citations": citations, "context": context}
