"""Normalize provider usage without retaining prompts or responses."""
from __future__ import annotations

from typing import Any, Dict

from .config import SETTINGS


def generation_badge(generation: Dict[str, Any]) -> str:
    quota_reasons = {
        "global_minute_limit", "global_day_limit", "session_minute_limit", "session_day_limit",
        "provider_quota",
    }
    if generation.get("fallback_reason") in quota_reasons:
        return "Quota fallback"
    if generation.get("mode") in {"groq_assisted_exact_evidence", "groq_structured"}:
        return "Groq-assisted"
    return "Exact extractive fallback"


def _value(source: Any, *names: str) -> int:
    for name in names:
        value = getattr(source, name, None)
        if value is None and isinstance(source, dict):
            value = source.get(name)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                pass
    return 0


def usage_from_response(response: Any) -> Dict[str, Any]:
    metadata = getattr(response, "usage_metadata", None) or {}
    input_tokens = _value(metadata, "prompt_token_count", "input_token_count")
    output_tokens = _value(metadata, "candidates_token_count", "output_token_count")
    cached_tokens = _value(metadata, "cached_content_token_count", "cached_token_count")
    total_tokens = _value(metadata, "total_token_count") or input_tokens + output_tokens
    estimated_cost = (
        input_tokens * SETTINGS.genai_input_cost_per_million_usd
        + output_tokens * SETTINGS.genai_output_cost_per_million_usd
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 8),
        "pricing_tier": SETTINGS.genai_pricing_tier,
    }
