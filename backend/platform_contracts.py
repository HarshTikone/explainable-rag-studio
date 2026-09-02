"""Stable ports used by local and production infrastructure adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, BinaryIO, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class StoredObject:
    organization_id: str
    object_key: str
    version_id: str | None
    checksum_sha256: str
    size_bytes: int
    encryption_algorithm: str


@dataclass(frozen=True)
class QueueReceipt:
    job_id: str
    queue_name: str


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    reset_after_seconds: int
    reason: str = ""


class ObjectStore(Protocol):
    def put(
        self,
        organization_id: str,
        source_name: str,
        content: bytes | BinaryIO,
        content_type: str,
        metadata: Mapping[str, str] | None = None,
    ) -> StoredObject: ...

    def get(self, organization_id: str, object_key: str) -> bytes: ...

    def soft_delete(self, organization_id: str, object_key: str) -> None: ...


class TaskQueue(Protocol):
    def enqueue(self, job_id: str) -> QueueReceipt: ...

    def cancel(self, job_id: str) -> bool: ...

    def health(self) -> Mapping[str, Any]: ...


class RateLimiter(Protocol):
    def check(self, key: str, limit: int, window_seconds: int, cost: int = 1) -> RateLimitDecision: ...


class VectorStore(Protocol):
    @property
    def meta(self) -> Mapping[str, Sequence[Mapping[str, Any]]]: ...

    def search(self, query_vec, top_k: int): ...


class SecretProvider(Protocol):
    def read(self, name: str) -> bytes: ...
