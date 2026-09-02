"""Produce a retained adversarial security-gate artifact."""
from __future__ import annotations

import json
import argparse
import os
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.security import SecurityRegistry
from backend.security_benchmark import security_benchmark, validate_security_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependency-passed", action="store_true")
    parser.add_argument("--static-passed", action="store_true")
    parser.add_argument("--api-passed", action="store_true")
    parser.add_argument("--streamlit-passed", action="store_true")
    parser.add_argument("--docker-passed", action="store_true")
    parser.add_argument("--platform-passed", action="store_true")
    parser.add_argument("--backup-restore-passed", action="store_true")
    args = parser.parse_args()
    cases = security_benchmark()
    counts = validate_security_benchmark(cases)
    tests = subprocess.run([
        sys.executable,
        "-m", "pytest", "-q", "tests/test_security.py", "tests/test_security_scanner.py",
        "tests/test_tenant_isolation.py", "tests/test_api_security.py",
    ], capture_output=True, text=True)
    # Windows can retain a SQLite handle briefly after WAL activity; cleanup is best-effort.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        registry = SecurityRegistry(str(Path(directory) / "security.db"), "benchmark-pepper", "benchmark-audit")
        boot = registry.bootstrap("Benchmark", "owner@benchmark.invalid")
        samples = []
        for _ in range(1000):
            started = time.perf_counter()
            registry.authenticate(boot["api_key"])
            samples.append((time.perf_counter() - started) * 1000)
        auth_p95 = sorted(samples)[949]
        audit_valid = registry.verify_audit_chain()["valid"]
    history = subprocess.run([sys.executable,
                              "scripts/scan_private_history.py"], capture_output=True, text=True)
    gates = {
        "adversarial_suite": tests.returncode == 0,
        "audit_chain": audit_valid,
        "auth_overhead_p95_under_50_ms": auth_p95 < 50,
        "private_history_clean": history.returncode == 0,
        "dependency_audit": args.dependency_passed,
        "static_analysis": args.static_passed,
        "fastapi_smoke": args.api_passed,
        "streamlit_smoke": args.streamlit_passed,
        "python_3_11": sys.version_info[:2] == (3, 11),
        "docker": args.docker_passed,
        "production_platform": args.platform_passed,
        "backup_restore": args.backup_restore_passed,
    }
    decision = "promoted" if all(gates.values()) else "rejected"
    output = {
        "schema_version": "2.0", "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "decision": decision, "category_counts": counts, "case_count": len(cases),
        "cases": [{**case, "result": "passed" if tests.returncode == 0 else "not_run"} for case in cases],
        "measured": {"authentication_p95_ms": auth_p95, "authentication_median_ms": statistics.median(samples)},
        "gates": gates,
        "failure_reasons": [name for name, passed in gates.items() if not passed],
        "test_output": (tests.stdout + tests.stderr)[-4000:], "history_output": (history.stdout + history.stderr)[-2000:],
    }
    target = Path("outputs/security")
    target.mkdir(parents=True, exist_ok=True)
    (target / "security_gate.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "gates": gates, "artifact": str(target / "security_gate.json")}, indent=2))


if __name__ == "__main__":
    main()
