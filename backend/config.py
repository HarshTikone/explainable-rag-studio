import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # chunking
    chunk_tokens: int = int(os.getenv("CHUNK_TOKENS", "420"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))

    # retrieval
    top_k: int = int(os.getenv("TOP_K", "6"))
    use_mmr: bool = os.getenv("USE_MMR", "true").lower() == "true"

    # embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Gemini generation
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")  # optional: SDK can also auto-pick from env
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # paths
    index_dir: str = "index"
    outputs_dir: str = "outputs"
    runs_db_path: str = os.path.join("outputs", "runs.db")

SETTINGS = Settings()
