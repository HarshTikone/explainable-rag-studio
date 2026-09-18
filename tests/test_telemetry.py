import sqlite3
from types import SimpleNamespace

from backend import generation_usage, observability, telemetry


def test_telemetry_additively_migrates_legacy_runs_table(tmp_path, monkeypatch):
    path = tmp_path / "runs.db"
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, query TEXT,
                top_k INTEGER, use_mmr INTEGER, retrieval_ms INTEGER,
                generation_ms INTEGER, total_ms INTEGER, citations TEXT
            )
        """)
    monkeypatch.setattr(telemetry, "SETTINGS", SimpleNamespace(outputs_dir=str(tmp_path), runs_db_path=str(path)))
    telemetry.log_run({
        "query": "q", "top_k": 6, "retrieval_ms": 10, "generation_ms": 20,
        "verification_ms": 5, "total_ms": 35, "grounding_status": "partial",
        "accepted_claims": 1, "rejected_claims": 2, "conflict_count": 1,
    })
    row = telemetry.fetch_runs()[0]
    assert row[8:] == (5, "partial", 1, 2, 1)
    with sqlite3.connect(path) as conn:
        stored = conn.execute(
            "SELECT query, query_sha256, query_length FROM runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert stored[0] == ""
    assert len(stored[1]) == 64
    assert stored[2] == 1


def test_token_cost_and_cache_metrics_are_aggregate_only(tmp_path, monkeypatch):
    path = tmp_path / "runs.db"
    settings = SimpleNamespace(
        outputs_dir=str(tmp_path), runs_db_path=str(path),
        genai_input_cost_per_million_usd=1.0,
        genai_output_cost_per_million_usd=2.0,
        genai_pricing_tier="configured",
    )
    monkeypatch.setattr(telemetry, "SETTINGS", settings)
    monkeypatch.setattr(generation_usage, "SETTINGS", settings)
    usage = generation_usage.usage_from_response(SimpleNamespace(usage_metadata=SimpleNamespace(
        prompt_token_count=100, candidates_token_count=50,
        cached_content_token_count=10, total_token_count=150,
    )))
    assert usage["estimated_cost_usd"] == 0.0002
    telemetry.record_cache_event("context", True)
    telemetry.record_cache_event("context", False)
    assert telemetry.fetch_cache_metrics()["context"] == {
        "hits": 1, "misses": 1, "hit_rate": 0.5,
    }


def test_otel_attribute_guard_drops_raw_and_identity_data():
    class Active:
        def __init__(self):
            self.values = {}

        def set_attribute(self, key, value):
            self.values[key] = value

    active = Active()
    observability.set_span_attributes(active, {
        "rag.query.sha256": "hash",
        "rag.query.length": 12,
        "rag.query.text": "secret question",
        "rag.evidence.text": "secret evidence",
        "enduser.user_id": "person",
        "client.ip_address": "127.0.0.1",
    })
    assert active.values == {"rag.query.sha256": "hash", "rag.query.length": 12}
