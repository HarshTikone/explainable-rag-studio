"""Lightweight external generation clients used by the hosted and full profiles."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict

import httpx


class ProviderRequestError(RuntimeError):
    """Sanitized provider failure that never includes prompts, evidence, or keys."""

    def __init__(self, status_code: int | None, category: str):
        self.status_code = status_code
        self.code = status_code
        self.category = category
        label = str(status_code) if status_code is not None else "transport"
        super().__init__(f"generation provider request failed ({label}:{category})")


@dataclass(frozen=True)
class ProviderUsageMetadata:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    cached_content_token_count: int = 0
    total_token_count: int = 0


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    parsed: Any = None
    usage_metadata: ProviderUsageMetadata = ProviderUsageMetadata()


def _strict_schema(model: Any) -> Dict[str, Any]:
    """Convert a Pydantic schema to Groq's strict structured-output subset."""
    schema = copy.deepcopy(model.model_json_schema())

    def normalize(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("default", None)
            if value.get("type") == "object" and isinstance(value.get("properties"), dict):
                value["required"] = list(value["properties"])
                value["additionalProperties"] = False
            for child in value.values():
                normalize(child)
        elif isinstance(value, list):
            for child in value:
                normalize(child)

    normalize(schema)
    return schema


class _GroqModels:
    def __init__(self, owner: "GroqClient"):
        self._owner = owner

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: Dict[str, Any] | None = None,
    ) -> ProviderResponse:
        return self._owner.generate_content(model=model, contents=contents, config=config)


class GroqClient:
    """Small compatibility adapter over Groq's OpenAI-compatible REST API."""

    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 12,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key.strip():
            raise ValueError("A Groq API key is required.")
        self._client = httpx.Client(
            timeout=timeout_seconds,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        self.models = _GroqModels(self)

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: Dict[str, Any] | None = None,
    ) -> ProviderResponse:
        config = config or {}
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": contents}],
            "temperature": config.get("temperature", 0),
            "max_completion_tokens": int(config.get("max_output_tokens", 256)),
            "stream": False,
        }
        if model.startswith("openai/gpt-oss-"):
            payload["reasoning_effort"] = "low"
        response_schema = config.get("response_schema")
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__.casefold(),
                    "strict": True,
                    "schema": _strict_schema(response_schema),
                },
            }
        try:
            response = self._client.post(self.endpoint, json=payload)
            response.raise_for_status()
            body = response.json()
            text = str(body["choices"][0]["message"].get("content") or "")
            usage = body.get("usage") or {}
            prompt_details = usage.get("prompt_tokens_details") or {}
            metadata = ProviderUsageMetadata(
                prompt_token_count=int(usage.get("prompt_tokens") or 0),
                candidates_token_count=int(usage.get("completion_tokens") or 0),
                cached_content_token_count=int(prompt_details.get("cached_tokens") or 0),
                total_token_count=int(usage.get("total_tokens") or 0),
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            category = {
                400: "invalid_request",
                401: "authentication",
                403: "permission",
                404: "model_not_found",
                429: "quota",
            }.get(status, "provider")
            raise ProviderRequestError(status, category) from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderRequestError(None, "transport_or_response") from exc

        parsed = None
        if response_schema is not None:
            try:
                parsed = response_schema.model_validate_json(text)
            except Exception:
                # The grounding layer owns validation/fallback semantics.
                parsed = None
        return ProviderResponse(text=text, parsed=parsed, usage_metadata=metadata)
