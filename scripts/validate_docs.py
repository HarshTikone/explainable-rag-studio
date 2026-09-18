"""Fail CI when portfolio claims drift from committed data/configuration."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import SETTINGS  # noqa: E402


def main() -> None:
    benchmark = json.loads((ROOT / "data" / "public_demo_benchmark.json").read_text(encoding="utf-8"))
    corpus_count = len(list((ROOT / "data" / "public_demo").glob("*.md")))
    categories = Counter(item.get("category") for item in benchmark)
    unanswerable_share = categories["unanswerable"] / max(1, len(benchmark))
    if corpus_count != 60 or len(benchmark) != 112 or len(categories) != 6 or unanswerable_share < 0.20:
        raise AssertionError("Documented benchmark shape no longer matches the committed dataset.")
    quality = json.loads(
        (ROOT / "docs" / "benchmarks" / "quality-gate-reference.json").read_text(encoding="utf-8")
    )["quality_gate"]
    measured = quality["grounding"]["gates"]["measured"]
    if not {"macro_f1", "supported_precision", "contradiction_recall"} <= set(measured):
        raise AssertionError("Committed grounding metrics are incomplete.")
    if quality["overall"].get("promoted") and not quality["grounding"].get("promoted"):
        raise AssertionError("Overall release cannot be promoted while grounding is rejected.")
    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    walkthrough = (ROOT / "docs" / "DEMO_WALKTHROUGH_SCRIPT.md").read_text(encoding="utf-8")
    results_page = (ROOT / "app" / "pages" / "9_Results.py").read_text(encoding="utf-8")
    for document in (readme, architecture, walkthrough):
        if SETTINGS.grounding_model not in document:
            raise AssertionError("Verifier model documentation drifted from configuration.")
    if "quality-gate-reference.json" not in results_page or "public_demo_benchmark.json" not in results_page:
        raise AssertionError("Results must render committed artifacts instead of copied metrics.")
    if SETTINGS.grounding_model != "cross-encoder/nli-deberta-v3-xsmall":
        raise AssertionError("Verifier model documentation drifted from configuration.")
    if not all(value in readme for value in (str(corpus_count), str(len(benchmark)), f"{unanswerable_share:.1%}")):
        raise AssertionError("README must report the current benchmark size and unanswerable share.")
    if "same 60 questions" in walkthrough:
        raise AssertionError("The walkthrough still claims the obsolete 60-question benchmark.")
    if "https://explainable-rag-studio-demo.onrender.com" not in readme:
        raise AssertionError("README must link to the public demo.")
    print(json.dumps({
        "corpus_documents": corpus_count,
        "benchmark_questions": len(benchmark),
        "category_counts": dict(sorted(categories.items())),
        "unanswerable_share": round(unanswerable_share, 3),
        "grounding_model": SETTINGS.grounding_model,
        "committed_grounding_metrics": measured,
        "overall_promoted": quality["overall"].get("promoted", False),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
