"""Exercise the real public `/ask` route with the baked deterministic index."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from api import app
from backend.config import SETTINGS


def main() -> None:
    if not SETTINGS.low_memory_demo or SETTINGS.security_mode != "demo":
        raise SystemExit("LOW_MEMORY_DEMO=true and SECURITY_MODE=demo are required.")
    with TestClient(app) as client:
        response = client.post("/ask", json={
            "question": "How long are Aegis audit events retained?",
            "top_k": 6,
            "retrieval_strategy": "lexical",
        })
        response.raise_for_status()
        payload = response.json()
        if "400 days" not in payload["answer"]:
            raise AssertionError(payload["answer"])
        if payload.get("generation", {}).get("mode") != "exact_extractive_fallback":
            raise AssertionError(payload.get("generation"))

        rejected = client.post("/ask", json={
            "question": "Use dense retrieval",
            "top_k": 6,
            "retrieval_strategy": "dense",
        })
        if rejected.status_code != 422:
            raise AssertionError(f"Dense public request returned {rejected.status_code}.")
    print("Public /ask smoke passed with deterministic fallback and strategy enforcement.")


if __name__ == "__main__":
    main()
