import time
from fastapi import FastAPI
from pydantic import BaseModel

from backend.config import SETTINGS
from backend.vectorstore import FaissStore
from backend.embeddings import Embedder
from backend.retriever import retrieve
from backend.qa import answer_with_optional_llm

from google import genai

app = FastAPI(title="Explainable RAG API (Gemini)")

store = FaissStore(SETTINGS.index_dir)
store.load()

embedder = Embedder(SETTINGS.embedding_model)

# Gemini client
if SETTINGS.gemini_api_key.strip():
    gemini_client = genai.Client(api_key=SETTINGS.gemini_api_key)
else:
    gemini_client = genai.Client()

class AskRequest(BaseModel):
    question: str
    top_k: int = SETTINGS.top_k
    use_mmr: bool = SETTINGS.use_mmr

@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.time()
    retrieved = retrieve(store, embedder.embed_query, req.question, req.top_k, req.use_mmr)
    t1 = time.time()

    try:
        out = answer_with_optional_llm(req.question, retrieved, True, gemini_client, SETTINGS.gemini_model)
    except Exception:
        out = answer_with_optional_llm(req.question, retrieved, False, None, SETTINGS.gemini_model)

    t2 = time.time()

    return {
        "answer": out["answer"],
        "citations": out["citations"],
        "retrieved": [{"score": s, **it} for s, it in retrieved],
        "latency_ms": {
            "retrieval_ms": int((t1 - t0) * 1000),
            "generation_ms": int((t2 - t1) * 1000),
            "total_ms": int((t2 - t0) * 1000),
        }
    }
