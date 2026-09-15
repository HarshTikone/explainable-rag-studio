"""Concurrent load test against a running API deployment.

Hits one authenticated, DB-backed endpoint with a fixed request budget spread across a thread
pool, and reports p50/p95/p99 latency, error rate, and achieved throughput. Stdlib-only so it
needs nothing beyond what's already installed to run the release-quality pipeline.

Requests round-robin across multiple caller identities (--api-keys, comma-separated) rather than
reusing one key for every request -- the platform's own per-identity rate limiter
(backend/rate_limit.py, default 60 GET requests/minute/identity) would otherwise dominate the
measurement at any real request volume, and multiple identities is a more realistic simulation of
concurrent users than one client hammering a single key anyway.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _one_request(url: str, api_key: str, timeout: float) -> tuple[float, int | None, str]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    started = time.perf_counter()
    try:
        # url is built from a caller-supplied --base-url/--endpoint pair, a fixed CI target, not
        # untrusted input reachable by an attacker.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            response.read()
            status = response.status
        return (time.perf_counter() - started) * 1000, status, ""
    except urllib.error.HTTPError as exc:
        return (time.perf_counter() - started) * 1000, exc.code, str(exc)
    except Exception as exc:  # noqa: BLE001 - any transport failure counts as a load-test error
        return (time.perf_counter() - started) * 1000, None, str(exc)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def run_load_test(
    base_url: str, endpoint: str, api_keys: list[str], total_requests: int, concurrency: int, timeout: float,
) -> dict:
    url = base_url.rstrip("/") + endpoint
    keys = itertools.islice(itertools.cycle(api_keys), total_requests)
    latencies_ms: list[float] = []
    statuses: dict[str, int] = {}
    errors: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one_request, url, key, timeout) for key in keys]
        for future in as_completed(futures):
            latency_ms, status, error = future.result()
            latencies_ms.append(latency_ms)
            key = str(status) if status is not None else "transport_error"
            statuses[key] = statuses.get(key, 0) + 1
            if error:
                errors.append(error)
    wall_seconds = time.perf_counter() - started
    successes = sum(count for status, count in statuses.items() if status.isdigit() and status.startswith("2"))
    return {
        "schema_version": "1.0",
        "endpoint": endpoint,
        "total_requests": total_requests,
        "concurrency": concurrency,
        "caller_identities": len(api_keys),
        "wall_seconds": wall_seconds,
        "throughput_rps": total_requests / wall_seconds if wall_seconds > 0 else 0.0,
        "status_counts": statuses,
        "success_rate": successes / total_requests if total_requests else 0.0,
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
            "p50": _percentile(latencies_ms, 0.50),
            "p95": _percentile(latencies_ms, 0.95),
            "p99": _percentile(latencies_ms, 0.99),
            "max": max(latencies_ms) if latencies_ms else 0.0,
        },
        "sample_errors": errors[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/documents")
    parser.add_argument("--api-keys", required=True, help="Comma-separated caller API keys, round-robined per request.")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", default="outputs/load_test/report.json")
    args = parser.parse_args()

    api_keys = [key.strip() for key in args.api_keys.split(",") if key.strip()]
    if not api_keys:
        raise SystemExit("--api-keys must contain at least one non-empty key.")
    report = run_load_test(args.base_url, args.endpoint, api_keys, args.requests, args.concurrency, args.timeout)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
