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
    conn.commit()
    conn.close()

def log_run(row: Dict[str, Any]):
    init_db()
    conn = sqlite3.connect(SETTINGS.runs_db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO runs (ts_ms, query, top_k, use_mmr, retrieval_ms, generation_ms, total_ms, citations)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row.get("ts_ms", now_ms()),
        row.get("query", ""),
        row.get("top_k", 0),
        1 if row.get("use_mmr", False) else 0,
        row.get("retrieval_ms", 0),
        row.get("generation_ms", 0),
        row.get("total_ms", 0),
        row.get("citations", ""),
    ))
    conn.commit()
    conn.close()

def fetch_runs(limit: int = 200):
    init_db()
    conn = sqlite3.connect(SETTINGS.runs_db_path)
    cur = conn.cursor()
    cur.execute("""
    SELECT ts_ms, query, top_k, use_mmr, retrieval_ms, generation_ms, total_ms, citations
    FROM runs
    ORDER BY id DESC
    LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows
