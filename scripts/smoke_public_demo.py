"""Cost-free end-to-end smoke for the 512 MB public image."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import SETTINGS
from backend.query_service import run_query
from backend.review_registry import ReviewRegistry
from backend.security_models import RetrievalScope
from backend.vectorstore import FaissStore


def main() -> None:
    if not SETTINGS.low_memory_demo:
        raise SystemExit("LOW_MEMORY_DEMO=true is required for the public smoke test.")
    store = FaissStore(SETTINGS.index_dir)
    if not store.load():
        raise SystemExit("The baked public index is unavailable.")
    scope = RetrievalScope(SETTINGS.public_organization_id, "release-smoke", "public-smoke")
    reviews = ReviewRegistry(SETTINGS.review_db_path)
    cases = (
        ("How long are Aegis audit events retained?", "400 days"),
        ("What incident ID investigated Meridian clock skew?", "ID-2026-014"),
        ("Are current Aegis audit events retained for 400 or 90 days?", "400 days"),
        ("What is the manufacturing cost of Northstar's hardware appliance?", "I don't know"),
    )
    results = []
    for question, expected in cases:
        value = run_query(
            store=store, question=question, top_k=min(6, SETTINGS.demo_top_k_max),
            strategy="lexical", scope=scope, review_registry=reviews,
            client_key="release-smoke", generation_client=None,
            organization_id=SETTINGS.public_organization_id,
            actor_user_id="release-smoke",
        )
        if expected.casefold() not in value["answer"].casefold():
            raise AssertionError(f"{question!r} did not contain {expected!r}: {value['answer']!r}")
        results.append({
            "question": question, "answer": value["answer"],
            "generation": value["generation"]["mode"],
            "latency_ms": value["latency_ms"]["total_ms"],
        })
    peak_rss_mb = None
    try:
        import resource
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if peak_rss_mb >= 450:
            raise AssertionError(f"Peak RSS {peak_rss_mb:.1f} MB exceeds the 450 MB release ceiling.")
    except ImportError:
        pass
    print(json.dumps({"passed": True, "peak_rss_mb": peak_rss_mb, "results": results}, indent=2))


if __name__ == "__main__":
    main()
