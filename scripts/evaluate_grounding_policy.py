"""Evaluate a previously locked policy against sealed held-out cases."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.grounding import CrossEncoderNliVerifier
from backend.grounding_eval import grounding_benchmark_fingerprint, run_grounding_benchmark
from backend.grounding_policy import GroundingPolicy
from backend.release_quality import load_public_corpus
from backend.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    parser.add_argument("--output", default="outputs/grounding_heldout.json")
    args = parser.parse_args()
    wrapper = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    policy = GroundingPolicy.from_dict(wrapper["policy"])
    cases = json.loads((ROOT / "data" / "grounding_benchmark.json").read_text(encoding="utf-8"))
    calibration = [case for case in cases if case["split"] == "calibration"]
    heldout = [case for case in cases if case["split"] == "heldout"]
    items = load_public_corpus(str(ROOT / "data" / "public_demo"))
    if grounding_benchmark_fingerprint(calibration, items) != wrapper["calibration_fingerprint"]:
        raise SystemExit("Calibration evidence changed after the policy was locked.")
    verifier = CrossEncoderNliVerifier(policy.model_name, policy.model_revision, policy.batch_size, policy.max_length)
    report = run_grounding_benchmark(heldout, items, verifier, policy)
    write_json(args.output, report)
    print(json.dumps({"output": args.output, "macro_f1": report["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
