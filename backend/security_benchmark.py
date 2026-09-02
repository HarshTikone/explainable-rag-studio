"""Deterministic adversarial benchmark specification and validation."""
from __future__ import annotations

CATEGORY_COUNTS = {
    "cross_tenant": 20, "bola": 10, "role_escalation": 10,
    "prompt_injection": 10, "upload_abuse": 10,
}


def security_benchmark() -> list[dict[str, str]]:
    cases = []
    offset = 0
    descriptions = {
        "cross_tenant": "Query overlapping identifiers and canary phrases across tenant boundaries",
        "bola": "Use an object identifier owned by another organization",
        "role_escalation": "Call an operation outside the role or reduced key scope",
        "prompt_injection": "Ingest or retrieve an indirect prompt-injection instruction",
        "upload_abuse": "Submit a mismatched, hidden, traversing, or over-expanded container",
    }
    for category, count in CATEGORY_COUNTS.items():
        for index in range(1, count + 1):
            offset += 1
            cases.append({"case_id": f"SEC-{offset:03d}", "category": category,
                          "description": f"{descriptions[category]} #{index}",
                          "expected": "blocked_without_sensitive_output"})
    return cases


def validate_security_benchmark(cases=None) -> dict[str, int]:
    cases = cases or security_benchmark()
    if len(cases) != 60 or len({case["case_id"] for case in cases}) != 60:
        raise ValueError("The security benchmark must contain 60 unique cases.")
    counts = {category: sum(case["category"] == category for case in cases) for category in CATEGORY_COUNTS}
    if counts != CATEGORY_COUNTS:
        raise ValueError(f"Security benchmark category mismatch: {counts}")
    return counts
