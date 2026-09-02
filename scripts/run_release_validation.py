"""Run or validate the complete release quality evidence pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.release_quality import (
    run_release_validation, update_release_runtime_checks, validate_release_artifact, write_portfolio_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", help="Validate an existing release directory instead of running models.")
    parser.add_argument("--portfolio-output", default="docs/benchmarks/quality-gate-reference.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--docker-passed", action="store_true")
    parser.add_argument("--streamlit-passed", action="store_true")
    parser.add_argument("--api-passed", action="store_true")
    parser.add_argument("--dependencies-passed", action="store_true")
    args = parser.parse_args()
    if args.validate:
        requested_checks = {
            "docker_build": args.docker_passed, "streamlit_health": args.streamlit_passed,
            "api_smoke": args.api_passed, "dependency_check": args.dependencies_passed,
        }
        if any(requested_checks.values()):
            update_release_runtime_checks(args.validate, requested_checks)
        result = validate_release_artifact(args.validate)
        result["portfolio_summary"] = write_portfolio_summary(args.validate, str(ROOT / args.portfolio_output))
        print(json.dumps(result, indent=2))
        return
    release_dir = run_release_validation(
        str(ROOT), repeats=max(1, args.repeats), runtime_checks={
            "docker_build": args.docker_passed, "streamlit_health": args.streamlit_passed,
            "api_smoke": args.api_passed, "dependency_check": args.dependencies_passed,
        },
    )
    summary = write_portfolio_summary(release_dir, str(ROOT / args.portfolio_output))
    print(json.dumps({"release_dir": release_dir, "portfolio_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
