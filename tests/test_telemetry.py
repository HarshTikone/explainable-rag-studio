import sqlite3
from types import SimpleNamespace

from backend import telemetry


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

