"""Small, privacy-safe OpenTelemetry bridge.

The application records spans everywhere, but exports nothing unless an OTLP
endpoint is explicitly configured. Attribute values passed here must be
operational metadata, never prompts, evidence text, secrets, or user IDs.
"""
from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import Any, Dict, Iterator

from .config import SETTINGS


_lock = Lock()
_configured = False
_SENSITIVE_KEY_PARTS = (
    ".prompt", ".evidence", ".api_key", ".user_id", ".user_identifier",
    ".ip", ".query.text", ".raw", ".document.text",
)


def _safe_attribute(key: str, value: Any) -> bool:
    lowered = key.casefold()
    return (
        isinstance(value, (str, bool, int, float))
        and not any(part in lowered for part in _SENSITIVE_KEY_PARTS)
    )


def _tracer():
    global _configured
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    if SETTINGS.otel_exporter_otlp_endpoint and not _configured:
        with _lock:
            if not _configured:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                    from opentelemetry.sdk.resources import Resource
                    from opentelemetry.sdk.trace import TracerProvider
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor

                    provider = TracerProvider(resource=Resource.create({
                        "service.name": SETTINGS.otel_service_name,
                    }))
                    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                        endpoint=SETTINGS.otel_exporter_otlp_endpoint,
                    )))
                    trace.set_tracer_provider(provider)
                except Exception:
                    # Telemetry must never make the query path unavailable.
                    pass
                _configured = True
    return trace.get_tracer(SETTINGS.otel_service_name)


@contextmanager
def span(name: str, attributes: Dict[str, Any] | None = None) -> Iterator[Any]:
    tracer = _tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as active:
        for key, value in (attributes or {}).items():
            if value is not None and _safe_attribute(key, value):
                active.set_attribute(key, value)
        yield active


def set_span_attributes(active, attributes: Dict[str, Any]) -> None:
    if active is None:
        return
    for key, value in attributes.items():
        if value is not None and _safe_attribute(key, value):
            active.set_attribute(key, value)
