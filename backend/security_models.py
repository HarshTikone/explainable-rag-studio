"""Security contracts shared by API, retrieval, ingestion, and UI layers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


PUBLIC_ORGANIZATION_ID = "org_public"
ROLES = ("owner", "admin", "editor", "viewer")
SCOPES = frozenset({
    "query", "documents:read", "documents:write", "reviews:write",
    "experiments:run", "security:read", "members:write",
})
ROLE_SCOPES = {
    "owner": SCOPES,
    "admin": frozenset(SCOPES - {"members:write"}),
    "editor": frozenset({"query", "documents:read", "documents:write"}),
    "viewer": frozenset({"query", "documents:read"}),
}


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


class SecurityBoundaryError(RuntimeError):
    pass


@dataclass(frozen=True)
class SecurityContext:
    organization_id: str
    user_id: str
    role: str
    key_id: str | None
    scopes: frozenset[str]
    anonymous_demo: bool = False

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"Unknown role: {self.role}")
        if not self.scopes <= SCOPES:
            raise ValueError("Security context contains unknown scopes.")

    def permits(self, scope: str) -> bool:
        return scope in ROLE_SCOPES[self.role] and scope in self.scopes

    def require(self, scope: str) -> None:
        if not self.permits(scope):
            raise AuthorizationError(f"Permission required: {scope}")


@dataclass(frozen=True)
class RetrievalScope:
    organization_id: str
    actor_user_id: str
    request_id: str = ""

    @classmethod
    def from_context(cls, context: SecurityContext, request_id: str = "") -> "RetrievalScope":
        context.require("query")
        return cls(context.organization_id, context.user_id, request_id)


@dataclass(frozen=True)
class SecurityFinding:
    code: str
    severity: str
    location: str = ""
    detector_version: str = "1.0"
    confidence: float = 1.0
    excerpt_hash: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "severity": self.severity, "location": self.location,
            "detector_version": self.detector_version, "confidence": self.confidence,
            "excerpt_hash": self.excerpt_hash, "details": dict(self.details),
        }
