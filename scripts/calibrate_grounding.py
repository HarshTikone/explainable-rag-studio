"""Calibrate and lock a grounding policy without reading held-out labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.grounding import CrossEncoderNliVerifier
from backend.grounding_eval import calibrate_grounding_policy
from backend.grounding_policy import GroundingPolicy, default_grounding_policy
from backend.release_quality import load_public_corpus
from backend.utils import write_json


def resolve_revision(model_name: str, fallback: str = "main") -> str:
    try:
        from huggingface_hub import model_info
        return str(model_info(model_name, revision=fallback).sha)
    except Exception:
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/grounding_policy.json")
    parser.add_argument("--allow-small-fallback", action="store_true")
    args = parser.parse_args()
    cases = json.loads((ROOT / "data" / "grounding_benchmark.json").read_text(encoding="utf-8"))
    calibration = [case for case in cases if case["split"] == "calibration"]
    items = load_public_corpus(str(ROOT / "data" / "public_demo"))
    base = default_grounding_policy()
    verifier = CrossEncoderNliVerifier(base.model_name, base.model_revision, base.batch_size, base.max_length)
    result = calibrate_grounding_policy(calibration, items, verifier, base)
    if not result["selected"] and args.allow_small_fallback:
        model = "cross-encoder/nli-deberta-v3-small"
        revision = resolve_revision(model)
        verifier = CrossEncoderNliVerifier(model, revision, base.batch_size, base.max_length)
        result = calibrate_grounding_policy(
            calibration, items, verifier, GroundingPolicy(model_name=model, model_revision=revision)
        )
    if not result["selected"]:
        raise SystemExit("No candidate met calibration safety constraints; no policy was locked.")
    wrapper = {
        "schema_version": "1.0", "locked": True,
        "calibration_fingerprint": result["calibration_fingerprint"],
        "policy": result["selected"]["policy"], "calibration": result,
    }
    write_json(args.output, wrapper)
    print(json.dumps({"output": args.output, "policy_id": wrapper["policy"]["policy_id"]}, indent=2))


if __name__ == "__main__":
    main()
