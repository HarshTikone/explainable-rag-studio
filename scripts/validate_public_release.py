"""Static release checks for the zero-cost Render profile."""
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    value = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    services = value.get("services", [])
    if len(services) != 1:
        raise AssertionError("The public Blueprint must contain exactly one service.")
    service = services[0]
    if service.get("plan") != "free":
        raise AssertionError("The public Render service must use plan: free.")
    if service.get("disk") or value.get("databases"):
        raise AssertionError("The public Blueprint cannot provision paid disks or databases.")
    if service.get("runtime") != "docker":
        raise AssertionError("The public service must use the Docker runtime.")
    env = {entry["key"]: entry.get("value") for entry in service.get("envVars", [])}
    required = {
        "LOW_MEMORY_DEMO": "true", "BUILD_DEMO_INDEX": "true",
        "PREFETCH_MODELS": "false", "PUBLIC_GEMINI_ENABLED": "true",
        "GENAI_PRICING_TIER": "free", "DEMO_GEMINI_GLOBAL_RPM": "2",
        "DEMO_GEMINI_GLOBAL_RPD": "20", "DEMO_GEMINI_SESSION_RPM": "1",
        "DEMO_GEMINI_SESSION_RPD": "5", "DEMO_GEMINI_TIMEOUT_SECONDS": "12",
        "DEMO_GEMINI_CIRCUIT_SECONDS": "300", "DEMO_QUESTION_MAX_CHARS": "500",
        "DEMO_CONTEXT_MAX_CHARS": "12000", "DEMO_TOP_K_MAX": "6",
        "GENAI_INPUT_COST_PER_MILLION_USD": "0",
        "GENAI_OUTPUT_COST_PER_MILLION_USD": "0",
    }
    for key, expected in required.items():
        if str(env.get(key, "")).lower() != expected:
            raise AssertionError(f"{key} must be {expected!r} in render.yaml.")
    if "GEMINI_API_KEY" not in env:
        raise AssertionError("GEMINI_API_KEY must be declared as a dashboard-managed secret.")
    public_requirements = (ROOT / "requirements-public.txt").read_text(encoding="utf-8").casefold()
    for heavyweight in ("sentence-transformers", "transformers==", "torch==", "onnxruntime"):
        if heavyweight in public_requirements:
            raise AssertionError(f"The public image cannot install heavyweight model runtime: {heavyweight}")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    if 'pip install -r requirements-public.txt' not in dockerfile:
        raise AssertionError("The low-memory Docker build must install requirements-public.txt.")
    print("Public Render release configuration is zero-cost and valid.")


if __name__ == "__main__":
    main()
