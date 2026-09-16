"""Durable, privacy-preserving Gemini allowance for the single-process demo."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock

from .config import SETTINGS


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str = ""


class DemoGeminiBudget:
    def __init__(self, path: str = SETTINGS.demo_budget_db_path):
        self.path = path
        self._lock = Lock()
        self.concurrency = BoundedSemaphore(1)
        self._circuit_until = 0.0

    def _connect(self):
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS demo_usage (
                scope TEXT NOT NULL,
                subject_hash TEXT NOT NULL,
                window_start INTEGER NOT NULL,
                used INTEGER NOT NULL,
                PRIMARY KEY (scope, subject_hash, window_start)
            )
        """)
        return connection

    @staticmethod
    def _subject(value: str) -> str:
        return hashlib.sha256((value or "anonymous").encode("utf-8")).hexdigest()

    def reserve(self, client_key: str, now: float | None = None) -> BudgetDecision:
        now = time.time() if now is None else now
        with self._lock:
            if now < self._circuit_until:
                return BudgetDecision(False, "provider_circuit_open")
            minute = int(now // 60 * 60)
            day = int(now // 86400 * 86400)
            subject = self._subject(client_key)
            checks = (
                ("global_minute", "global", minute, SETTINGS.demo_gemini_global_rpm),
                ("global_day", "global", day, SETTINGS.demo_gemini_global_rpd),
                ("session_minute", subject, minute, SETTINGS.demo_gemini_session_rpm),
                ("session_day", subject, day, SETTINGS.demo_gemini_session_rpd),
            )
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM demo_usage WHERE window_start < ?", (day - 86400,))
                for scope, owner, window_start, limit in checks:
                    row = connection.execute(
                        "SELECT used FROM demo_usage WHERE scope=? AND subject_hash=? AND window_start=?",
                        (scope, owner, window_start),
                    ).fetchone()
                    if limit <= 0 or (row and int(row[0]) >= limit):
                        connection.rollback()
                        return BudgetDecision(False, scope + "_limit")
                for scope, owner, window_start, _ in checks:
                    connection.execute("""
                        INSERT INTO demo_usage(scope, subject_hash, window_start, used)
                        VALUES (?, ?, ?, 1)
                        ON CONFLICT(scope, subject_hash, window_start)
                        DO UPDATE SET used=used+1
                    """, (scope, owner, window_start))
                connection.commit()
            return BudgetDecision(True)

    def open_circuit(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._circuit_until = max(self._circuit_until, now + SETTINGS.demo_gemini_circuit_seconds)

    def reset_circuit(self) -> None:
        with self._lock:
            self._circuit_until = 0.0


DEMO_GEMINI_BUDGET = DemoGeminiBudget()
