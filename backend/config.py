import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # production platform (legacy keeps the offline FAISS/SQLite demo readable)
    platform_mode: str = os.getenv("PLATFORM_MODE", "legacy").strip().lower()
    database_url: str = os.getenv("DATABASE_URL", "")
    # Bypasses row-level security for pg_dump/pg_restore; database_url's role
    # (rag_app) is deliberately NOBYPASSRLS and can't produce a complete backup.
    # Falls back to database_url where an admin connection isn't configured.
    admin_database_url: str = os.getenv("ADMIN_DATABASE_URL", "") or os.getenv("DATABASE_URL", "")
    redis_url: str = os.getenv("REDIS_URL", "")
    object_store_endpoint: str = os.getenv("OBJECT_STORE_ENDPOINT", "")
    object_store_region: str = os.getenv("OBJECT_STORE_REGION", "us-east-1")
    object_store_bucket: str = os.getenv("OBJECT_STORE_BUCKET", "rag-private")
    object_store_access_key: str = os.getenv("OBJECT_STORE_ACCESS_KEY", "")
    object_store_secret_key: str = os.getenv("OBJECT_STORE_SECRET_KEY", "")
    object_store_secure: bool = os.getenv("OBJECT_STORE_SECURE", "true").lower() == "true"
    object_master_key_file: str = os.getenv("OBJECT_MASTER_KEY_FILE", "")
    oidc_issuer: str = os.getenv("OIDC_ISSUER", "")
    oidc_audience: str = os.getenv("OIDC_AUDIENCE", "explainable-rag-api")
    oidc_client_id: str = os.getenv("OIDC_CLIENT_ID", "explainable-rag-streamlit")
    oidc_client_secret_file: str = os.getenv("OIDC_CLIENT_SECRET_FILE", "")
    oidc_redirect_uri: str = os.getenv("OIDC_REDIRECT_URI", "http://localhost:8000/auth/callback")
    app_public_url: str = os.getenv("APP_PUBLIC_URL", "http://localhost:8501")
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    oidc_jwks_ttl_seconds: int = int(os.getenv("OIDC_JWKS_TTL_SECONDS", "300"))
    rate_limit_query_per_minute: int = int(os.getenv("RATE_LIMIT_QUERY_PER_MINUTE", "60"))
    rate_limit_write_per_minute: int = int(os.getenv("RATE_LIMIT_WRITE_PER_MINUTE", "20"))
    rate_limit_ingest_per_hour: int = int(os.getenv("RATE_LIMIT_INGEST_PER_HOUR", "10"))
    quota_documents: int = int(os.getenv("QUOTA_DOCUMENTS", "1000"))
    quota_upload_bytes: int = int(os.getenv("QUOTA_UPLOAD_BYTES", str(5 * 1024 * 1024 * 1024)))
    quota_concurrent_jobs: int = int(os.getenv("QUOTA_CONCURRENT_JOBS", "4"))
    audit_retention_days: int = int(os.getenv("AUDIT_RETENTION_DAYS", "365"))
    release_retention_days: int = int(os.getenv("RELEASE_RETENTION_DAYS", "90"))
    deleted_retention_days: int = int(os.getenv("DELETED_RETENTION_DAYS", "30"))

    # security and tenant isolation
    security_mode: str = os.getenv("SECURITY_MODE", "required").strip().lower()
    security_db_path: str = os.getenv("SECURITY_DB_PATH", os.path.join("outputs", "security.db"))
    api_key_pepper: str = os.getenv("API_KEY_PEPPER", "")
    audit_hmac_key: str = os.getenv("AUDIT_HMAC_KEY", "")
    public_organization_id: str = os.getenv("PUBLIC_ORGANIZATION_ID", "org_public")
    api_key_expiry_days: int = int(os.getenv("API_KEY_EXPIRY_DAYS", "90"))
    tenant_store_cache_size: int = int(os.getenv("TENANT_STORE_CACHE_SIZE", "8"))
    security_policy_version: str = os.getenv("SECURITY_POLICY_VERSION", "1.0")
    max_archive_entries: int = int(os.getenv("MAX_ARCHIVE_ENTRIES", "500"))
    max_decompressed_mb: int = int(os.getenv("MAX_DECOMPRESSED_MB", "100"))
    max_compression_ratio: float = float(os.getenv("MAX_COMPRESSION_RATIO", "100"))
    max_document_pages: int = int(os.getenv("MAX_DOCUMENT_PAGES", "1000"))

    # chunking
    chunk_tokens: int = int(os.getenv("CHUNK_TOKENS", "420"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    parent_tokens: int = int(os.getenv("PARENT_TOKENS", "1200"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    max_documents_per_job: int = int(os.getenv("MAX_DOCUMENTS_PER_JOB", "10"))
    ocr_executable: str = os.getenv("OCR_EXECUTABLE", "")
    context_mode: str = os.getenv("CONTEXT_MODE", "deterministic")
    context_model: str = os.getenv("CONTEXT_MODEL", "gemini-2.5-flash")
    context_prompt_version: str = os.getenv("CONTEXT_PROMPT_VERSION", "1.0")
    ingestion_worker_lease_seconds: int = int(os.getenv("INGESTION_WORKER_LEASE_SECONDS", "120"))
    ingestion_retry_limit: int = int(os.getenv("INGESTION_RETRY_LIMIT", "3"))

    # retrieval
    top_k: int = int(os.getenv("TOP_K", "6"))
    use_mmr: bool = os.getenv("USE_MMR", "true").lower() == "true"
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "30"))
    rerank_batch_size: int = int(os.getenv("RERANK_BATCH_SIZE", "16"))
    reranker_backend: str = os.getenv("RERANKER_BACKEND", "onnx").strip().lower()
    reranker_model_revision: str = os.getenv("RERANKER_MODEL_REVISION", "c5f2b386de279a97c53a702dd5189d1c407160dc")
    reranker_onnx_file: str = os.getenv("RERANKER_ONNX_FILE", "onnx/model_O1.onnx")
    model_cpu_threads: int = int(os.getenv("MODEL_CPU_THREADS", "2"))

    # claim-level grounding
    grounding_model: str = os.getenv("GROUNDING_MODEL", "cross-encoder/nli-deberta-v3-xsmall")
    grounding_model_revision: str = os.getenv("GROUNDING_MODEL_REVISION", "a150876415327c80daeff35ca6f68f5ed8cf5c24")
    grounding_batch_size: int = int(os.getenv("GROUNDING_BATCH_SIZE", "16"))
    grounding_max_length: int = int(os.getenv("GROUNDING_MAX_LENGTH", "512"))
    grounding_evidence_scan_k: int = int(os.getenv("GROUNDING_EVIDENCE_SCAN_K", "8"))
    grounding_max_claims: int = int(os.getenv("GROUNDING_MAX_CLAIMS", "8"))
    grounding_entailment_threshold: float = float(os.getenv("GROUNDING_ENTAILMENT_THRESHOLD", "0.75"))
    grounding_contradiction_threshold: float = float(os.getenv("GROUNDING_CONTRADICTION_THRESHOLD", "0.70"))
    grounding_low_confidence_margin: float = float(os.getenv("GROUNDING_LOW_CONFIDENCE_MARGIN", "0.10"))
    grounding_prompt_version: str = os.getenv("GROUNDING_PROMPT_VERSION", "1.0")
    grounding_conflict_relevance_threshold: float = float(os.getenv("GROUNDING_CONFLICT_RELEVANCE_THRESHOLD", "0.30"))
    grounding_premise_strategy: str = os.getenv("GROUNDING_PREMISE_STRATEGY", "context_envelope")
    grounding_backend: str = os.getenv("GROUNDING_BACKEND", "onnx").strip().lower()
    grounding_onnx_file: str = os.getenv("GROUNDING_ONNX_FILE", "onnx/model_quint8_avx2.onnx")
    grounding_fallback_model: str = os.getenv("GROUNDING_FALLBACK_MODEL", "cross-encoder/nli-deberta-v3-small")
    grounding_fallback_revision: str = os.getenv("GROUNDING_FALLBACK_REVISION", "fa2804872c3b4bd748f38c0185cc85775361e735")

    # embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Gemini generation
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")  # optional: SDK can also auto-pick from env
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # paths
    index_dir: str = "index"
    outputs_dir: str = "outputs"
    runs_db_path: str = os.path.join("outputs", "runs.db")
    ingestion_db_path: str = os.path.join("outputs", "ingestion.db")
    review_db_path: str = os.path.join("outputs", "reviews.db")
    uploads_dir: str = os.path.join("outputs", "uploads")
    tenant_index_root: str = os.path.join("index", "organizations")

SETTINGS = Settings()
