"""PostgreSQL security registry with RLS-safe credential routing."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from sqlalchemy import func, select, text

from .database import (
    ApiKey, AuditEvent, DatabaseRuntime, GlobalApiKeyLookup, GlobalIdentityLookup,
    Membership, Organization, QuarantineCase, User,
)
from .oidc import OidcPrincipal
from .security import KEY_PATTERN, _canonical, _clean, _redact_details
from .security_models import AuthenticationError, AuthorizationError, ROLE_SCOPES, ROLES, SCOPES, SecurityContext


class PostgresSecurityRegistry:
    def __init__(self, database: DatabaseRuntime, api_key_pepper: str, audit_hmac_key: str):
        self.database = database
        self.api_key_pepper = api_key_pepper.encode()
        self.audit_hmac_key = audit_hmac_key.encode()

    @property
    def protected_ready(self) -> bool:
        return bool(self.api_key_pepper and self.audit_hmac_key)

    def require_protected_configuration(self) -> None:
        if not self.protected_ready:
            raise RuntimeError("API_KEY_PEPPER and AUDIT_HMAC_KEY are required in protected mode.")

    @staticmethod
    def _system_context(organization_id: str, user_id: str = "system") -> SecurityContext:
        return SecurityContext(organization_id, user_id, "owner", None, ROLE_SCOPES["owner"])

    def ensure_public_organization(self, organization_id: str = "org_public") -> None:
        with self.database.engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO organizations (organization_id,name,active,created_at) "
                "VALUES (:id,'Public Demo',true,now()) ON CONFLICT (organization_id) DO NOTHING"
            ), {"id": organization_id})

    def _key_digest(self, key_id: str, secret: str) -> str:
        if not self.api_key_pepper:
            raise RuntimeError("API_KEY_PEPPER is not configured.")
        return hmac.new(self.api_key_pepper, f"{key_id}:{secret}".encode(), hashlib.sha256).hexdigest()

    def bootstrap(self, organization_name: str, owner_email: str, owner_name: str = "Owner") -> dict[str, str]:
        self.require_protected_configuration()
        organization_id, user_id = "org_" + uuid.uuid4().hex[:16], "usr_" + uuid.uuid4().hex[:16]
        with self.database.engine.begin() as connection:
            if connection.execute(select(func.count()).select_from(Organization).where(Organization.organization_id != "org_public")).scalar_one():
                raise ValueError("A first owner already exists; use membership management instead.")
            connection.execute(Organization.__table__.insert().values(organization_id=organization_id, name=_clean(organization_name, 160), active=True))
            connection.execute(User.__table__.insert().values(user_id=user_id, email=owner_email.casefold(), display_name=_clean(owner_name, 160), active=True))
            connection.execute(text("SELECT set_config('app.organization_id', :value, true)"), {"value": organization_id})
            connection.execute(Membership.__table__.insert().values(organization_id=organization_id, user_id=user_id, role="owner"))
        result = self.create_api_key(organization_id, user_id, ROLE_SCOPES["owner"])
        self.append_audit(organization_id, user_id, result["key_id"], "security.bootstrap", "organization", organization_id, "success", 201)
        return {"organization_id": organization_id, "user_id": user_id, **result}

    def create_api_key(self, organization_id: str, user_id: str, scopes: Iterable[str], expires_days: int = 90) -> dict[str, str]:
        selected = frozenset(scopes)
        if not selected <= SCOPES:
            raise ValueError("Unknown API-key scope.")
        context = self._system_context(organization_id, user_id)
        key_id, secret = secrets.token_hex(6), secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
        with self.database.session(context, write=True) as session:
            membership = session.get(Membership, (organization_id, user_id))
            if not membership:
                raise KeyError("Membership not found.")
            if not selected <= ROLE_SCOPES[membership.role]:
                raise AuthorizationError("API-key scopes cannot exceed role permissions.")
            session.add(ApiKey(key_id=key_id, organization_id=organization_id, user_id=user_id,
                               digest=self._key_digest(key_id, secret), scopes=sorted(selected), expires_at=expires_at))
            session.add(GlobalApiKeyLookup(key_id=key_id, organization_id=organization_id))
        return {"key_id": key_id, "api_key": f"ragk_{key_id}_{secret}", "expires_at": expires_at.isoformat()}

    def create_key_for_context(self, context: SecurityContext, scopes: Iterable[str], expires_days: int = 90) -> dict[str, str]:
        if context.role not in {"owner", "admin"}:
            raise AuthorizationError("Only owners and admins can create keys.")
        if not frozenset(scopes) <= context.scopes:
            raise AuthorizationError("A new key cannot exceed the caller's scopes.")
        result = self.create_api_key(context.organization_id, context.user_id, scopes, expires_days)
        self.append_audit(context.organization_id, context.user_id, context.key_id, "api_key.create", "api_key", result["key_id"], "success", 201)
        return result

    def authenticate(self, raw_key: str) -> SecurityContext:
        match = KEY_PATTERN.fullmatch((raw_key or "").strip())
        if not match:
            raise AuthenticationError("Invalid credentials.")
        key_id, secret = match.groups()
        with self.database.engine.connect() as connection:
            lookup = connection.execute(select(GlobalApiKeyLookup.organization_id).where(GlobalApiKeyLookup.key_id == key_id)).first()
        organization_id = lookup.organization_id if lookup else "org_invalid"
        context = self._system_context(organization_id)
        with self.database.session(context, write=True) as session:
            row = session.execute(
                select(ApiKey, Membership.role, Organization.active, User.active)
                .join(Membership, (Membership.organization_id == ApiKey.organization_id) & (Membership.user_id == ApiKey.user_id))
                .join(Organization, Organization.organization_id == ApiKey.organization_id)
                .join(User, User.user_id == ApiKey.user_id)
                .where(ApiKey.key_id == key_id)
            ).first()
            supplied = self._key_digest(key_id, secret)
            stored = row[0].digest if row else "0" * 64
            if (not row or not hmac.compare_digest(supplied, stored) or row[0].revoked_at
                    or not row[2] or not row[3] or row[0].expires_at <= datetime.now(timezone.utc)):
                raise AuthenticationError("Invalid credentials.")
            row[0].last_used_at = datetime.now(timezone.utc)
            scopes = frozenset(row[0].scopes) & ROLE_SCOPES[row.role]
            return SecurityContext(row[0].organization_id, row[0].user_id, row.role, key_id, scopes)

    def register_oidc_identity(self, context: SecurityContext, user_id: str, issuer: str, subject: str) -> None:
        context.require("members:write")
        identity_hash = hashlib.sha256(f"{issuer.rstrip('/')}\x1f{subject}".encode()).hexdigest()
        with self.database.session(context, write=True) as session:
            if session.get(Membership, (context.organization_id, user_id)) is None:
                raise KeyError("Membership not found.")
            row = session.get(GlobalIdentityLookup, (identity_hash, context.organization_id))
            if row and row.user_id != user_id:
                raise ValueError("OIDC identity is already assigned.")
            if not row:
                session.add(GlobalIdentityLookup(issuer_subject_hash=identity_hash, organization_id=context.organization_id, user_id=user_id))

    def authenticate_oidc(self, principal: OidcPrincipal) -> SecurityContext:
        identity_hash = hashlib.sha256(f"{principal.issuer.rstrip('/')}\x1f{principal.subject}".encode()).hexdigest()
        with self.database.engine.connect() as connection:
            mappings = connection.execute(select(GlobalIdentityLookup).where(GlobalIdentityLookup.issuer_subject_hash == identity_hash)).scalars().all()
        requested = str(principal.claims.get("organization_id", ""))
        if requested:
            mappings = [mapping for mapping in mappings if mapping.organization_id == requested]
        if len(mappings) != 1:
            raise AuthenticationError("OIDC identity does not resolve to exactly one organization.")
        mapping = mappings[0]
        system = self._system_context(mapping.organization_id, mapping.user_id)
        with self.database.session(system) as session:
            row = session.execute(
                select(Membership.role, User.active, Organization.active)
                .join(User, User.user_id == Membership.user_id)
                .join(Organization, Organization.organization_id == Membership.organization_id)
                .where(Membership.user_id == mapping.user_id)
            ).first()
        if not row or not row[1] or not row[2]:
            raise AuthenticationError("OIDC membership is inactive.")
        return SecurityContext(mapping.organization_id, mapping.user_id, row.role, None, ROLE_SCOPES[row.role])

    def list_api_keys(self, context: SecurityContext) -> list[dict[str, Any]]:
        if context.role not in {"owner", "admin"}:
            raise AuthorizationError("Key administration is restricted.")
        with self.database.session(context) as session:
            rows = session.execute(select(ApiKey).order_by(ApiKey.created_at.desc())).scalars().all()
            return [{"key_id": row.key_id, "user_id": row.user_id, "scopes": row.scopes,
                     "expires_at": row.expires_at.isoformat(), "revoked_at": row.revoked_at.isoformat() if row.revoked_at else None,
                     "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None} for row in rows]

    def revoke_api_key(self, context: SecurityContext, key_id: str) -> None:
        if context.role not in {"owner", "admin"}:
            raise AuthorizationError("Key administration is restricted.")
        with self.database.session(context, write=True) as session:
            row = session.get(ApiKey, key_id)
            if not row or row.organization_id != context.organization_id:
                raise KeyError("API key not found.")
            row.revoked_at = datetime.now(timezone.utc)

    def add_user(self, context: SecurityContext, email: str, display_name: str, role: str) -> str:
        context.require("members:write")
        if context.role != "owner" or role not in ROLES:
            raise AuthorizationError("Only owners can manage memberships.")
        user_id = "usr_" + uuid.uuid4().hex[:16]
        with self.database.session(context, write=True) as session:
            session.add(User(user_id=user_id, email=email.casefold(), display_name=_clean(display_name, 160), active=True))
            session.add(Membership(organization_id=context.organization_id, user_id=user_id, role=role))
        return user_id

    def list_members(self, context: SecurityContext) -> list[dict[str, Any]]:
        context.require("members:write")
        with self.database.session(context) as session:
            rows = session.execute(select(User.user_id, User.email, User.display_name, Membership.role, Membership.created_at)
                                   .join(Membership, Membership.user_id == User.user_id).order_by(User.email)).all()
            return [dict(row._mapping) for row in rows]

    def change_membership(self, context: SecurityContext, user_id: str, role: str | None) -> None:
        context.require("members:write")
        if context.role != "owner" or (role is not None and role not in ROLES):
            raise AuthorizationError("Only owners can manage memberships.")
        with self.database.session(context, write=True) as session:
            membership = session.get(Membership, (context.organization_id, user_id))
            if not membership:
                raise KeyError("Membership not found.")
            if membership.role == "owner" and role != "owner":
                owners = session.scalar(select(func.count()).select_from(Membership).where(Membership.role == "owner"))
                if owners <= 1:
                    raise ValueError("The final organization owner cannot be removed or demoted.")
            if role is None:
                session.delete(membership)
            else:
                membership.role = role

    def append_audit(self, organization_id: str, user_id: str | None, key_id: str | None, action: str,
                     object_type: str, object_id: str | None, result: str, http_status: int,
                     reason_code: str = "", request_id: str = "", details: dict[str, Any] | None = None) -> str:
        if not self.audit_hmac_key:
            raise RuntimeError("AUDIT_HMAC_KEY is not configured.")
        context = self._system_context(organization_id, user_id or "system")
        event_id, created_at = "aud_" + uuid.uuid4().hex, datetime.now(timezone.utc)
        with self.database.session(context, write=True) as session:
            session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:value))"), {"value": organization_id})
            previous = session.execute(select(AuditEvent.event_hash).order_by(AuditEvent.sequence.desc()).limit(1)).scalar_one_or_none() or "GENESIS"
            payload = {"event_id": event_id, "organization_id": _clean(organization_id, 80), "user_id": _clean(user_id, 80),
                       "key_id": _clean(key_id, 80), "action": _clean(action, 128), "object_type": _clean(object_type, 64),
                       "object_id": _clean(object_id, 128), "result": _clean(result, 32), "reason_code": _clean(reason_code, 64),
                       "http_status": int(http_status), "request_id": _clean(request_id, 96), "details": _redact_details(details),
                       "previous_hash": previous, "created_at": created_at.isoformat()}
            event_hash = hmac.new(self.audit_hmac_key, _canonical(payload).encode(), hashlib.sha256).hexdigest()
            session.add(AuditEvent(event_id=event_id, organization_id=organization_id, user_id=user_id, key_id=key_id,
                                   action=payload["action"], object_type=payload["object_type"], object_id=payload["object_id"],
                                   result=payload["result"], reason_code=payload["reason_code"], http_status=http_status,
                                   request_id=payload["request_id"], details_json=payload["details"], previous_hash=previous,
                                   event_hash=event_hash, created_at=created_at))
        return event_id

    def list_audit(self, context: SecurityContext, limit: int = 100) -> list[dict[str, Any]]:
        context.require("security:read")
        with self.database.session(context) as session:
            rows = session.execute(select(AuditEvent).order_by(AuditEvent.sequence.desc()).limit(min(max(limit, 1), 1000))).scalars().all()
            return [{column.name: getattr(row, column.name) for column in AuditEvent.__table__.columns} for row in rows]

    def verify_audit_chain(self, context: SecurityContext | None = None) -> dict[str, Any]:
        if context is None:
            raise AuthorizationError("A tenant context is required for audit verification.")
        rows = list(reversed(self.list_audit(context, 1000)))
        previous = "GENESIS"
        for row in rows:
            payload = {"event_id": row["event_id"], "organization_id": row["organization_id"], "user_id": _clean(row["user_id"], 80),
                       "key_id": _clean(row["key_id"], 80), "action": row["action"], "object_type": row["object_type"],
                       "object_id": _clean(row["object_id"], 128), "result": row["result"], "reason_code": row["reason_code"],
                       "http_status": row["http_status"], "request_id": row["request_id"], "details": row["details_json"],
                       "previous_hash": previous, "created_at": row["created_at"].isoformat()}
            expected = hmac.new(self.audit_hmac_key, _canonical(payload).encode(), hashlib.sha256).hexdigest()
            if row["previous_hash"] != previous or not hmac.compare_digest(expected, row["event_hash"]):
                return {"valid": False, "events": len(rows), "failed_event_id": row["event_id"]}
            previous = row["event_hash"]
        return {"valid": True, "events": len(rows), "head": previous}

    def create_quarantine_case(self, organization_id: str, document_name: str, findings: list[dict[str, Any]], created_by: str, staged_path: str = "", job_id: str = "") -> str:
        case_id = "qtn_" + uuid.uuid4().hex
        context = self._system_context(organization_id, created_by)
        with self.database.session(context, write=True) as session:
            session.add(QuarantineCase(organization_id=organization_id, case_id=case_id, job_id=job_id or None,
                                       document_name=_clean(document_name, 512), object_key=staged_path,
                                       findings_json=findings, created_by=created_by))
        return case_id

    def list_quarantine(self, context: SecurityContext, status: str | None = None) -> list[dict[str, Any]]:
        if context.role not in {"owner", "admin"}:
            raise AuthorizationError("Quarantine access is restricted.")
        with self.database.session(context) as session:
            query = select(QuarantineCase)
            if status:
                query = query.where(QuarantineCase.status == status)
            rows = session.execute(query.order_by(QuarantineCase.created_at.desc())).scalars().all()
            return [self._quarantine_dict(row, internal=False) for row in rows]

    def get_quarantine_internal(self, context: SecurityContext, case_id: str) -> dict[str, Any] | None:
        if context.role not in {"owner", "admin"}:
            raise AuthorizationError("Quarantine access is restricted.")
        with self.database.session(context) as session:
            row = session.get(QuarantineCase, (context.organization_id, case_id))
            return self._quarantine_dict(row, internal=True) if row else None

    def resolve_quarantine(self, context: SecurityContext, case_id: str, decision: str, reason: str) -> None:
        if context.role not in {"owner", "admin"} or decision not in {"approved", "rejected"} or not reason.strip():
            raise AuthorizationError("A reasoned owner/admin quarantine decision is required.")
        with self.database.session(context, write=True) as session:
            row = session.get(QuarantineCase, (context.organization_id, case_id))
            if not row or row.status != "open":
                raise KeyError("Quarantine case not found.")
            row.status, row.resolved_by, row.resolution_reason = decision, context.user_id, _clean(reason, 1000)
            row.resolved_at = datetime.now(timezone.utc)

    @staticmethod
    def _quarantine_dict(row: QuarantineCase, internal: bool) -> dict[str, Any]:
        value = {"case_id": row.case_id, "organization_id": row.organization_id, "job_id": row.job_id,
                 "document_name": row.document_name, "status": row.status, "findings": row.findings_json,
                 "created_by": row.created_by, "created_at": row.created_at.isoformat(), "resolved_by": row.resolved_by,
                 "resolution_reason": row.resolution_reason, "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None}
        if internal:
            value["staged_path"] = row.object_key
        return value
