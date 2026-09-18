import hashlib
import sqlite3
from typing import Any, Dict
from .config import SETTINGS
from .utils import ensure_dir, now_ms

def init_db():
    ensure_dir(SETTINGS.outputs_dir)
    conn = sqlite3.connect(SETTINGS.runs_db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_ms INTEGER,
        query TEXT,
        top_k INTEGER,
        use_mmr INTEGER,
        retrieval_ms INTEGER,
        generation_ms INTEGER,
        total_ms INTEGER,
        citations TEXT
    )
    """)
    existing = {row[1] for row in cur.execute("PRAGMA table_info(runs)").fetchall()}
    additions = {
        "verification_ms": "INTEGER NOT NULL DEFAULT 0",
        "grounding_status": "TEXT NOT NULL DEFAULT ''",
        "accepted_claims": "INTEGER NOT NULL DEFAULT 0",
        "rejected_claims": "INTEGER NOT NULL DEFAULT 0",
        "conflict_count": "INTEGER NOT NULL DEFAULT 0",
        "organization_id": "TEXT NOT NULL DEFAULT 'org_public'",
        "actor_user_id": "TEXT NOT NULL DEFAULT ''",
        "query_sha256": "TEXT NOT NULL DEFAULT ''",
        "query_length": "INTEGER NOT NULL DEFAULT 0",
        "query_category": "TEXT NOT NULL DEFAULT ''",
        "generation_mode": "TEXT NOT NULL DEFAULT ''",
        "generation_fallback_reason": "TEXT NOT NULL DEFAULT ''",
        "input_tokens": "INTEGER NOT NULL DEFAULT 0",
        "output_tokens": "INTEGER NOT NULL DEFAULT 0",
        "cached_tokens": "INTEGER NOT NULL DEFAULT 0",
        "total_tokens": "INTEGER NOT NULL DEFAULT 0",
        "estimated_cost_usd": "REAL NOT NULL DEFAULT 0",
        "pricing_tier": "TEXT NOT NULL DEFAULT ''",
    }
    for column, definition in additions.items():
        if column not in existing:
            cur.execute(f"ALTER TABLE runs ADD COLUMN {column} {definition}")
    conn.commit()
    conn.close()

def log_run(row: Dict[str, Any]):
    init_db()
    conn = sqlite3.connect(SETTINGS.runs_db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO runs (
        ts_ms, query, top_k, use_mmr, retrieval_ms, generation_ms, total_ms, citations,
        verification_ms, grounding_status, accepted_claims, rejected_claims, conflict_count,
        organization_id, actor_user_id, query_sha256, query_length, query_category,
        generation_mode, generation_fallback_reason, input_tokens, output_tokens,
        cached_tokens, total_tokens, estimated_cost_usd, pricing_tier
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row.get("ts_ms", now_ms()),
        "",  # deprecated: never persist raw query text
        row.get("top_k", 0),
        1 if row.get("use_mmr", False) else 0,
        row.get("retrieval_ms", 0),
        row.get("generation_ms", 0),
        row.get("total_ms", 0),
        str(row.get("citation_count", 0)),  # deprecated column retains count only, never citation content
        row.get("verification_ms", 0),
        row.get("grounding_status", ""),
        row.get("accepted_claims", 0),
        row.get("rejected_claims", 0),
        row.get("conflict_count", 0),
        row.get("organization_id", "org_public"),
        row.get("actor_user_id", ""),
        hashlib.sha256(row.get("query", "").encode("utf-8")).hexdigest() if row.get("query") else row.get("query_sha256", ""),
        len(row.get("query", "")) if row.get("query") else row.get("query_length", 0),
        row.get("query_category", ""),
        row.get("generation_mode", ""),
        row.get("generation_fallback_reason", ""),
        row.get("input_tokens", 0),
        row.get("output_tokens", 0),
        row.get("cached_tokens", 0),
        row.get("total_tokens", 0),
        row.get("estimated_cost_usd", 0.0),
        row.get("pricing_tier", ""),
    ))
    conn.commit()
    conn.close()

def fetch_runs(limit: int = 200):
    init_db()
    conn = sqlite3.connect(SETTINGS.runs_db_path)
    cur = conn.cursor()
    cur.execute("""
    SELECT ts_ms, query, top_k, use_mmr, retrieval_ms, generation_ms, total_ms, citations,
           verification_ms, grounding_status, accepted_claims, rejected_claims, conflict_count
    FROM runs
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows


def record_cache_event(cache_name: str, hit: bool) -> None:
    """Aggregate cache outcomes without storing keys or cached content."""
    ensure_dir(SETTINGS.outputs_dir)
    with sqlite3.connect(SETTINGS.runs_db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_metrics (
                cache_name TEXT PRIMARY KEY,
                hits INTEGER NOT NULL DEFAULT 0,
                misses INTEGER NOT NULL DEFAULT 0
            )
        """)
        if hit:
            conn.execute(
                "INSERT INTO cache_metrics(cache_name, hits) VALUES (?, 1) "
                "ON CONFLICT(cache_name) DO UPDATE SET hits=hits+1",
                (cache_name,),
            )
        else:
            conn.execute(
                "INSERT INTO cache_metrics(cache_name, misses) VALUES (?, 1) "
                "ON CONFLICT(cache_name) DO UPDATE SET misses=misses+1",
                (cache_name,),
            )


def fetch_cache_metrics() -> Dict[str, Dict[str, float]]:
    ensure_dir(SETTINGS.outputs_dir)
    with sqlite3.connect(SETTINGS.runs_db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_metrics (
                cache_name TEXT PRIMARY KEY,
                hits INTEGER NOT NULL DEFAULT 0,
                misses INTEGER NOT NULL DEFAULT 0
            )
        """)
        rows = conn.execute("SELECT cache_name, hits, misses FROM cache_metrics ORDER BY cache_name").fetchall()
    return {
        name: {"hits": hits, "misses": misses, "hit_rate": hits / max(1, hits + misses)}
        for name, hits, misses in rows
    }
