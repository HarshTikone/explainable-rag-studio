from typing import List, Dict, Any
import tiktoken

def _get_encoder():
    # 'cl100k_base' works well for GPT-style tokenization.
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    enc = _get_encoder()
    return len(enc.encode(text))

def chunk_text_token_based(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int
) -> List[str]:
    """
    Splits raw text into chunks using token counts.
    Keeps overlap for better cross-boundary retrieval.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
        start = max(0, end - overlap_tokens)
    return chunks

def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_tokens: int,
    overlap_tokens: int
) -> List[Dict[str, Any]]:
    """
    Converts PDF pages into chunk objects with metadata.
    """
    all_chunks = []
    chunk_id = 0

    for p in pages:
        page_text = p["text"]
        pieces = chunk_text_token_based(page_text, chunk_tokens, overlap_tokens)

        for piece in pieces:
            chunk_id += 1
            all_chunks.append({
                "chunk_id": f"c{chunk_id:06d}",
                "source": p["source"],
                "page": p["page"],
                "text": piece,
                "token_count": count_tokens(piece),
            })

    return all_chunks
