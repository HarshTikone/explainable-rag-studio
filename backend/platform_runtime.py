"""Composition root for the opt-in portable production platform."""
from __future__ import annotations

from functools import lru_cache

from .config import SETTINGS
from .database import DatabaseRuntime, GlobalJobLookup
from .distributed_jobs import RqTaskQueue
from .object_store import S3EnvelopeObjectStore
from .oidc import OidcValidator, PkceStateStore
from .postgres_store import PgVectorStore
from .production_ingestion import ProductionIngestionService
from .rate_limit import RedisRateLimiter
from .secrets import FileSecretProvider
from .security_models import ROLE_SCOPES, SecurityContext


class PlatformConfigurationError(RuntimeError):
    pass


class PlatformRuntime:
    def __init__(self):
        missing = [name for name, value in {
            "DATABASE_URL": SETTINGS.database_url,
            "REDIS_URL": SETTINGS.redis_url,
            "OBJECT_STORE_ENDPOINT": SETTINGS.object_store_endpoint,
            "OBJECT_STORE_ACCESS_KEY": SETTINGS.object_store_access_key,
            "OBJECT_STORE_SECRET_KEY": SETTINGS.object_store_secret_key,
            "OBJECT_MASTER_KEY_FILE": SETTINGS.object_master_key_file,
            "OIDC_ISSUER": SETTINGS.oidc_issuer,
        }.items() if not value]
        if missing:
            raise PlatformConfigurationError("Production platform configuration is incomplete: " + ", ".join(missing))
        import redis

        self.database = DatabaseRuntime(SETTINGS.database_url)
        self.redis = redis.Redis.from_url(SETTINGS.redis_url, decode_responses=False)
        secrets = FileSecretProvider({"object_master_key": SETTINGS.object_master_key_file})
        self.object_store = S3EnvelopeObjectStore(
            SETTINGS.object_store_endpoint,
            SETTINGS.object_store_region,
            SETTINGS.object_store_bucket,
            SETTINGS.object_store_access_key,
            SETTINGS.object_store_secret_key,
            secrets,
            secure=SETTINGS.object_store_secure,
        )
        self.rate_limiter = RedisRateLimiter(self.redis)
        self.queue = RqTaskQueue(self.redis)
        self.ingestion = ProductionIngestionService(self.database, self.object_store, self.queue)
        self.oidc = OidcValidator(SETTINGS.oidc_issuer, SETTINGS.oidc_audience, SETTINGS.oidc_jwks_ttl_seconds)
        self.pkce = PkceStateStore(self.redis)

    def vector_store(self, context: SecurityContext) -> PgVectorStore:
        return PgVectorStore(self.database, context)

    def process_ingestion_job(self, job_id: str) -> None:
        from sqlalchemy import select

        with self.database.engine.connect() as connection:
            row = connection.execute(select(GlobalJobLookup.organization_id, GlobalJobLookup.created_by).where(GlobalJobLookup.job_id == job_id)).first()
        if not row:
            raise KeyError("Ingestion job not found.")
        context = SecurityContext(row.organization_id, row.created_by, "editor", None, ROLE_SCOPES["editor"])
        self.ingestion.process(context, job_id)

    def health(self) -> dict:
        checks = {"postgres": False, "redis": False, "object_store": False, "queue": False, "oidc": False}
        try:
            checks["postgres"] = self.database.ping()
        except Exception:
            pass
        try:
            checks["redis"] = bool(self.redis.ping())
            checks["queue"] = bool(self.queue.health().get("ready"))
        except Exception:
            pass
        try:
            self.object_store.client.head_bucket(Bucket=self.object_store.bucket)
            checks["object_store"] = True
        except Exception:
            pass
        try:
            self.oidc._refresh()
            checks["oidc"] = True
        except Exception:
            pass
        return {"ready": all(checks.values()), "checks": checks, "storage": "postgres_pgvector", "isolation": "postgres_rls"}


@lru_cache(maxsize=1)
def get_platform_runtime(required: bool = False) -> PlatformRuntime | None:
    if SETTINGS.platform_mode != "postgres":
        if required:
            raise PlatformConfigurationError("PLATFORM_MODE=postgres is required for this worker.")
        return None
    return PlatformRuntime()
