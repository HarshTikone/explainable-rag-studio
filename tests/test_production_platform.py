import io
import json
from types import SimpleNamespace

import pytest

from backend.database import DatabaseRuntime, TENANT_TABLES, rls_statements
from backend.grounding import verify_claims
from backend.grounding_models import DraftClaim, StructuredDraft
from backend.object_store import EnvelopeCipher, MemoryEnvelopeObjectStore, ObjectIntegrityError, S3EnvelopeObjectStore
from backend.oidc import PkceStateStore
from backend.platform_operations import _libpq_url
from backend.rate_limit import MemoryDemoRateLimiter
from backend.secrets import FileSecretProvider


class FakeRedis:
    def __init__(self):
        self.values = {}

    def setex(self, key, _ttl, value):
        self.values[key] = value

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis):
        self.redis, self.key = redis, None

    def get(self, key):
        self.key = key
        return self

    def delete(self, _key):
        return self

    def execute(self):
        return [self.redis.values.pop(self.key, None), 1]


class FakeValidator:
    authorization_endpoint = "https://identity.example/authorize"


class FakeS3:
    def __init__(self):
        self.values = {}

    def put_object(self, **kwargs):
        self.values[kwargs["Key"]] = {"Body": bytes(kwargs["Body"]), "Metadata": kwargs["Metadata"]}
        return {"VersionId": "v1"}

    def get_object(self, Bucket, Key):
        row = self.values[Key]
        return {"Body": io.BytesIO(row["Body"]), "Metadata": row["Metadata"]}

    def delete_object(self, Bucket, Key):
        self.values.pop(Key, None)


def test_rls_is_forced_for_every_tenant_table():
    statements = rls_statements()
    for table in TENANT_TABLES:
        assert f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY' in statements
        assert f'ALTER TABLE "{table}" FORCE ROW LEVEL SECURITY' in statements
        policy = next(item for item in statements if item.startswith(f'CREATE POLICY "{table}_tenant_isolation"'))
        assert "current_setting('app.organization_id'" in policy
        assert "WITH CHECK" in policy


def test_database_runtime_rejects_non_postgres_urls():
    with pytest.raises(ValueError):
        DatabaseRuntime("sqlite:///unsafe.db")


def test_libpq_url_strips_sqlalchemy_driver_suffix():
    assert _libpq_url("postgresql+psycopg://rag_app:pw@postgres:5432/rag") == "postgresql://rag_app:pw@postgres:5432/rag"
    assert _libpq_url("postgresql://rag_app:pw@postgres:5432/rag") == "postgresql://rag_app:pw@postgres:5432/rag"


def test_envelope_cipher_authenticates_tenant_and_ciphertext():
    cipher = EnvelopeCipher(b"k" * 32)
    value = cipher.encrypt("org_alpha", b"private")
    assert cipher.decrypt("org_alpha", value) == b"private"
    with pytest.raises(ObjectIntegrityError):
        cipher.decrypt("org_beta", value)
    damaged = value[:-1] + bytes([value[-1] ^ 1])
    with pytest.raises(ObjectIntegrityError):
        cipher.decrypt("org_alpha", damaged)


def test_s3_store_uses_random_tenant_keys_and_rejects_cross_tenant(tmp_path):
    secret = tmp_path / "master"
    secret.write_bytes(b"m" * 32)
    s3 = FakeS3()
    store = S3EnvelopeObjectStore("", "us-east-1", "bucket", "id", "secret",
                                  FileSecretProvider({"object_master_key": str(secret)}), client=s3)
    saved = store.put("org_alpha", "report.pdf", b"content", "application/pdf")
    assert saved.object_key.startswith("organizations/org_alpha/uploads/")
    assert store.get("org_alpha", saved.object_key) == b"content"
    with pytest.raises(PermissionError):
        store.get("org_beta", saved.object_key)


def test_demo_limiter_is_bounded():
    limiter = MemoryDemoRateLimiter()
    assert limiter.check("client", 2, 60).allowed
    assert limiter.check("client", 2, 60).allowed
    decision = limiter.check("client", 2, 60)
    assert not decision.allowed
    assert decision.remaining == 0


def test_pkce_state_is_one_time_and_contains_no_verifier_in_url():
    redis = FakeRedis()
    store = PkceStateStore(redis)
    started = store.begin(FakeValidator(), "client", "https://app.example/callback")
    assert "code_challenge=" in started["authorization_url"]
    assert "verifier" not in started["authorization_url"]
    pending = store.consume(started["state"])
    assert pending["verifier"]
    with pytest.raises(Exception):
        store.consume(started["state"])


def test_exact_extractive_claim_skips_nli_but_remains_supported():
    class NoNli:
        model_name = "none"
        model_revision = "none"

        def score(self, pairs):
            assert pairs == []
            return []

    draft = StructuredDraft(answerable=True, claims=[DraftClaim(
        text="The current procedure uses TS-999.", cited_chunk_ids=["c1"], provenance="extractive"
    )])
    result = verify_claims(draft, [(1.0, {"chunk_id": "c1", "text": "The current procedure uses TS-999.",
                                                   "generation_text": "The current procedure uses TS-999.",
                                                   "source": "card.md"})], verifier=NoNli()).result
    assert result.status == "verified"
    assert result.accepted_claims[0].entailment_score == 1.0
