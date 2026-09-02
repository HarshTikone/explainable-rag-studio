"""Run the real local verifier against the reproducible grounding benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import SETTINGS
from backend.contextual_chunking import build_contextual_chunks
from backend.document_parsers import parse_document
from backend.grounding import get_default_verifier
from backend.grounding_eval import compare_grounding_policies, grounding_promotion_gate, run_grounding_benchmark, save_grounding_artifact


def load_corpus():
    chunks = []
    for path in sorted((ROOT / "data" / "public_demo").glob("*.md")):
        parsed = parse_document(str(path), source_name=path.name)
        chunks.extend(build_contextual_chunks(parsed.document, parsed.version, parsed.blocks))
    return [chunk.to_dict() for chunk in chunks]


def load_qa_report(path_value: str):
    path = Path(path_value)
    report = json.loads(path.read_text(encoding="utf-8"))
    results_path = path.with_name("results.json")
    if "results" not in report and results_path.exists():
        report["results"] = json.loads(results_path.read_text(encoding="utf-8"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-qa-report", help="Schema 3.3 strict QA report JSON")
    parser.add_argument("--baseline-qa-report", help="Matching unfiltered/baseline QA report JSON")
    args = parser.parse_args()
    cases = json.loads((ROOT / "data" / "grounding_benchmark.json").read_text(encoding="utf-8"))
    report = run_grounding_benchmark(cases, load_corpus(), get_default_verifier())
    gate = {"passed": False, "reason": "Matching baseline and strict QA reports were not supplied."}
    policy_comparison = {}
    if args.strict_qa_report and args.baseline_qa_report:
        strict = load_qa_report(args.strict_qa_report)
        baseline = load_qa_report(args.baseline_qa_report)
        gate = grounding_promotion_gate(report, strict, baseline)
        policy_comparison = compare_grounding_policies(strict)
    path = save_grounding_artifact(report, gate, SETTINGS.outputs_dir, policy_comparison)
    print(json.dumps({"artifact": path, "n": report["n"], "macro_f1": report["macro_f1"], "promotion": gate}, indent=2))


if __name__ == "__main__":
    main()
