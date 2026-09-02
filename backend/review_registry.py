"""Durable, local human-review queue for uncertain grounding decisions."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ReviewRegistry:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_cases (
                    case_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'open',
                    reason TEXT NOT NULL,
                    claim_text TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS review_cases_status_idx
                    ON review_cases(status, created_at DESC);
                CREATE TABLE IF NOT EXISTS review_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES review_cases(case_id)
                );
                """
            )

    def enqueue(self, payload: Dict[str, Any], reason: str, config_fingerprint: str) -> str:
        identity = {
            "claim": payload.get("text", ""),
            "evidence": [item.get("chunk_id", "") for item in payload.get("evidence", [])],
            "config": config_fingerprint,
        }
        case_id = "rev_" + hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:20]
        timestamp = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO review_cases (
                    case_id, status, reason, claim_text, verdict, payload_json,
                    config_fingerprint, created_at, updated_at
                ) VALUES (?, 'open', ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    reason=excluded.reason, verdict=excluded.verdict,
                    payload_json=excluded.payload_json, updated_at=excluded.updated_at
                """,
                (
                    case_id, reason, payload.get("text", ""), payload.get("verdict", "unsupported"),
                    _canonical(payload), config_fingerprint, timestamp, timestamp,
                ),
            )
        return case_id

    def list_cases(self, status: str = "open", limit: int = 100) -> List[Dict[str, Any]]:
        query = "SELECT * FROM review_cases"
        params: List[Any] = []
        if status != "all":
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        with self._connect() as conn:
            return [self._decode(row) for row in conn.execute(query, params).fetchall()]

    def get_case(self, case_id: str) -> Dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM review_cases WHERE case_id = ?", (case_id,)).fetchone()
            if not row:
                return None
            result = self._decode(row)
            result["decisions"] = [dict(item) for item in conn.execute(
                "SELECT decision, reviewer, notes, created_at FROM review_decisions WHERE case_id = ? ORDER BY id",
                (case_id,),
            ).fetchall()]
            return result

    def decide(self, case_id: str, decision: str, reviewer: str = "local", notes: str = "") -> bool:
        if decision not in {"supported", "unsupported", "contradicted"}:
            raise ValueError("Unknown review decision.")
        timestamp = _now()
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM review_cases WHERE case_id = ?", (case_id,)).fetchone()
            if not exists:
                return False
            conn.execute(
                "INSERT INTO review_decisions(case_id, decision, reviewer, notes, created_at) VALUES (?, ?, ?, ?, ?)",
                (case_id, decision, reviewer.strip() or "local", notes.strip(), timestamp),
            )
            conn.execute(
                "UPDATE review_cases SET status = 'resolved', updated_at = ? WHERE case_id = ?",
                (timestamp, case_id),
            )
        return True

    def export_jsonl(self, status: str = "all") -> str:
        cases = self.list_cases(status=status, limit=500)
        detailed = [self.get_case(case["case_id"]) for case in cases]
        return "\n".join(_canonical(case) for case in detailed if case is not None)

    @staticmethod
    def _decode(row: sqlite3.Row) -> Dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result
