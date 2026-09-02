"""SQLite-backed authentication, authorization metadata, quarantine, and audit."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .security_models import (
    AuthenticationError, AuthorizationError, ROLE_SCOPES, ROLES, SCOPES,
    SecurityContext,
)

KEY_PATTERN = re.compile(r"^ragk_([A-Za-z0-9]{12})_([A-Za-z0-9_-]{43})$")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SENSITIVE_KEYS = {"api_key", "authorization", "secret", "prompt", "answer", "text", "path", "evidence"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any, limit: int = 500) -> str:
    return CONTROL_PATTERN.sub("", str(value or ""))[:limit]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _redact_details(details: dict[str, Any] | None) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in (details or {}).items():
        name = _clean(key, 80)
        if name.casefold() in SENSITIVE_KEYS or any(token in name.casefold() for token in ("secret", "token", "password")):
            cleaned[name] = "[REDACTED]"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[name] = _clean(value, 500) if isinstance(value, str) else value
        else:
            cleaned[name] = _clean(_canonical(value), 500)
    return cleaned


class SecurityRegistry:
    def __init__(self, db_path: str, api_key_pepper: str = "", audit_hmac_key: str = ""):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.api_key_pepper = api_key_pepper.encode("utf-8")
        self.audit_hmac_key = audit_hmac_key.encode("utf-8")
        self._write_lock = threading.RLock()
        self._initialize()

    @property
    def protected_ready(self) -> bool:
        return bool(self.api_key_pepper and self.audit_hmac_key)

    def require_protected_configuration(self) -> None:
        if not self.protected_ready:
            raise RuntimeError("API_KEY_PEPPER and AUDIT_HMAC_KEY are required in protected mode.")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS organizations (
                    organization_id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memberships (
                    organization_id TEXT NOT NULL, user_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('owner','admin','editor','viewer')),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (organization_id, user_id),
                    FOREIGN KEY (organization_id) REFERENCES organizations(organization_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY, organization_id TEXT NOT NULL, user_id TEXT NOT NULL,
                    digest TEXT NOT NULL, scopes_json TEXT NOT NULL, expires_at TEXT NOT NULL,
                    revoked_at TEXT, last_used_at TEXT, created_at TEXT NOT NULL,
                    FOREIGN KEY (organization_id, user_id) REFERENCES memberships(organization_id, user_id)
                );
                CREATE TABLE IF NOT EXISTS quarantine_cases (
                    case_id TEXT PRIMARY KEY, organization_id TEXT NOT NULL, job_id TEXT,
                    document_name TEXT NOT NULL, trust_state TEXT NOT NULL DEFAULT 'quarantined',
                    status TEXT NOT NULL DEFAULT 'open', findings_json TEXT NOT NULL,
                    staged_path TEXT, created_by TEXT NOT NULL, created_at TEXT NOT NULL,
                    resolved_by TEXT, resolution_reason TEXT, resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS quarantine_org_idx
                    ON quarantine_cases(organization_id, status, created_at);
                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
                    organization_id TEXT NOT NULL, user_id TEXT, key_id TEXT,
                    action TEXT NOT NULL, object_type TEXT NOT NULL, object_id TEXT,
                    result TEXT NOT NULL, reason_code TEXT, http_status INTEGER NOT NULL,
                    request_id TEXT, details_json TEXT NOT NULL, previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS audit_org_idx
                    ON audit_events(organization_id, sequence);
                CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_events
                BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_events
                BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END;
            """)

    def bootstrap(self, organization_name: str, owner_email: str, owner_name: str = "Owner") -> dict[str, str]:
        self.require_protected_configuration()
        organization_id = "org_" + uuid.uuid4().hex[:16]
        user_id = "usr_" + uuid.uuid4().hex[:16]
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM memberships LIMIT 1").fetchone():
                raise ValueError("A first owner already exists; use membership management instead.")
            connection.execute("INSERT INTO organizations VALUES (?, ?, 1, ?)", (organization_id, _clean(organization_name, 120), _now()))
            connection.execute("INSERT INTO users VALUES (?, ?, ?, 1, ?)", (user_id, owner_email.strip().casefold(), _clean(owner_name, 120), _now()))
            connection.execute("INSERT INTO memberships VALUES (?, ?, 'owner', ?)", (organization_id, user_id, _now()))
        key = self.create_api_key(organization_id, user_id, ROLE_SCOPES["owner"])
        self.append_audit(organization_id, user_id, key["key_id"], "security.bootstrap", "organization", organization_id, "success", 201)
        return {"organization_id": organization_id, "user_id": user_id, **key}

    def ensure_public_organization(self, organization_id: str = "org_public") -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO organizations VALUES (?, 'Public Demo', 1, ?)",
                (organization_id, _now()),
            )

    def add_user(self, context: SecurityContext, email: str, display_name: str, role: str) -> str:
        context.require("members:write")
        if context.role != "owner" or role not in ROLES:
            raise AuthorizationError("Only owners can manage memberships.")
        user_id = "usr_" + uuid.uuid4().hex[:16]
        with self._connect() as connection:
            connection.execute("INSERT INTO users VALUES (?, ?, ?, 1, ?)", (user_id, email.strip().casefold(), _clean(display_name, 120), _now()))
            connection.execute("INSERT INTO memberships VALUES (?, ?, ?, ?)", (context.organization_id, user_id, role, _now()))
        return user_id

    def list_members(self, context: SecurityContext) -> list[dict[str, Any]]:
        context.require("members:write")
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT u.user_id,u.email,u.display_name,m.role,m.created_at
                FROM memberships m JOIN users u ON u.user_id=m.user_id
                WHERE m.organization_id=? ORDER BY u.email
            """, (context.organization_id,)).fetchall()
        return [dict(row) for row in rows]

    def change_membership(self, context: SecurityContext, user_id: str, role: str | None) -> None:
        context.require("members:write")
        if context.role != "owner" or (role is not None and role not in ROLES):
            raise AuthorizationError("Only owners can manage memberships.")
        with self._connect() as connection:
            current = connection.execute("SELECT role FROM memberships WHERE organization_id=? AND user_id=?", (context.organization_id, user_id)).fetchone()
            if not current:
                raise KeyError("Membership not found.")
            if current["role"] == "owner" and role != "owner":
                owners = connection.execute("SELECT COUNT(*) FROM memberships WHERE organization_id=? AND role='owner'", (context.organization_id,)).fetchone()[0]
                if owners <= 1:
                    raise ValueError("The final organization owner cannot be removed or demoted.")
            if role is None:
                connection.execute("DELETE FROM memberships WHERE organization_id=? AND user_id=?", (context.organization_id, user_id))
            else:
                connection.execute("UPDATE memberships SET role=? WHERE organization_id=? AND user_id=?", (role, context.organization_id, user_id))

    def _key_digest(self, key_id: str, secret: str) -> str:
        if not self.api_key_pepper:
            raise RuntimeError("API_KEY_PEPPER is not configured.")
        return hmac.new(self.api_key_pepper, f"{key_id}:{secret}".encode(), hashlib.sha256).hexdigest()

    def create_api_key(self, organization_id: str, user_id: str, scopes: Iterable[str], expires_days: int = 90) -> dict[str, str]:
        selected = frozenset(scopes)
        if not selected <= SCOPES:
            raise ValueError("Unknown API-key scope.")
        key_id = secrets.token_hex(6)
        secret = secrets.token_urlsafe(32)
        digest = self._key_digest(key_id, secret)
        expiry = (datetime.now(timezone.utc) + timedelta(days=expires_days)).isoformat()
        with self._connect() as connection:
            membership = connection.execute("SELECT role FROM memberships WHERE organization_id=? AND user_id=?", (organization_id, user_id)).fetchone()
            if not membership:
                raise KeyError("Membership not found.")
            allowed = ROLE_SCOPES[membership["role"]]
            if not selected <= allowed:
                raise AuthorizationError("API-key scopes cannot exceed role permissions.")
            connection.execute(
                "INSERT INTO api_keys VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?)",
                (key_id, organization_id, user_id, digest, _canonical(sorted(selected)), expiry, _now()),
            )
        return {"key_id": key_id, "api_key": f"ragk_{key_id}_{secret}", "expires_at": expiry}

    def create_key_for_context(self, context: SecurityContext, scopes: Iterable[str], expires_days: int = 90) -> dict[str, str]:
        if context.role not in {"owner", "admin"}:
            raise AuthorizationError("Only owners and admins can create keys.")
        selected = frozenset(scopes)
        if not selected <= context.scopes:
            raise AuthorizationError("A new key cannot exceed the caller's scopes.")
        result = self.create_api_key(context.organization_id, context.user_id, selected, expires_days)
        self.append_audit(context.organization_id, context.user_id, context.key_id, "api_key.create", "api_key", result["key_id"], "success", 201)
        return result

    def authenticate(self, raw_key: str) -> SecurityContext:
        match = KEY_PATTERN.fullmatch((raw_key or "").strip())
        if not match:
            raise AuthenticationError("Invalid credentials.")
        key_id, secret = match.groups()
        with self._connect() as connection:
            row = connection.execute("""
                SELECT k.*,m.role,o.active AS org_active,u.active AS user_active
                FROM api_keys k JOIN memberships m ON m.organization_id=k.organization_id AND m.user_id=k.user_id
                JOIN organizations o ON o.organization_id=k.organization_id
                JOIN users u ON u.user_id=k.user_id WHERE k.key_id=?
            """, (key_id,)).fetchone()
            supplied = self._key_digest(key_id, secret)
            stored = row["digest"] if row else "0" * 64
            valid = hmac.compare_digest(supplied, stored)
            if not row or not valid or row["revoked_at"] or not row["org_active"] or not row["user_active"]:
                raise AuthenticationError("Invalid credentials.")
            if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
                raise AuthenticationError("Invalid credentials.")
            connection.execute("UPDATE api_keys SET last_used_at=? WHERE key_id=?", (_now(), key_id))
        scopes = frozenset(json.loads(row["scopes_json"])) & ROLE_SCOPES[row["role"]]
        return SecurityContext(row["organization_id"], row["user_id"], row["role"], key_id, scopes)

    def list_api_keys(self, context: SecurityContext) -> list[dict[str, Any]]:
        if context.role not in {"owner", "admin"}:
            raise AuthorizationError("Key administration is restricted.")
        with self._connect() as connection:
            rows = connection.execute("SELECT key_id,user_id,scopes_json,expires_at,revoked_at,last_used_at,created_at FROM api_keys WHERE organization_id=? ORDER BY created_at DESC", (context.organization_id,)).fetchall()
        result = []
        for row in rows:
            value = dict(row); value["scopes"] = json.loads(value.pop("scopes_json")); result.append(value)
        return result

    def revoke_api_key(self, context: SecurityContext, key_id: str) -> None:
        if context.role not in {"owner", "admin"}:
            raise AuthorizationError("Key administration is restricted.")
        with self._connect() as connection:
            cursor = connection.execute("UPDATE api_keys SET revoked_at=? WHERE key_id=? AND organization_id=? AND revoked_at IS NULL", (_now(), key_id, context.organization_id))
            if not cursor.rowcount:
                raise KeyError("API key not found.")
        self.append_audit(context.organization_id, context.user_id, context.key_id, "api_key.revoke", "api_key", key_id, "success", 200)

    def append_audit(self, organization_id: str, user_id: str | None, key_id: str | None,
                     action: str, object_type: str, object_id: str | None, result: str,
                     http_status: int, reason_code: str = "", request_id: str = "",
                     details: dict[str, Any] | None = None) -> str:
        if not self.audit_hmac_key:
            raise RuntimeError("AUDIT_HMAC_KEY is not configured.")
        event_id, created_at = "aud_" + uuid.uuid4().hex, _now()
        redacted = _redact_details(details)
        with self._write_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1").fetchone()
            previous_hash = previous[0] if previous else "GENESIS"
            payload = {
                "event_id": event_id, "organization_id": _clean(organization_id, 80),
                "user_id": _clean(user_id, 80), "key_id": _clean(key_id, 80),
                "action": _clean(action, 120), "object_type": _clean(object_type, 80),
                "object_id": _clean(object_id, 160), "result": _clean(result, 40),
                "reason_code": _clean(reason_code, 80), "http_status": int(http_status),
                "request_id": _clean(request_id, 100), "details": redacted,
                "previous_hash": previous_hash, "created_at": created_at,
            }
            event_hash = hmac.new(self.audit_hmac_key, _canonical(payload).encode(), hashlib.sha256).hexdigest()
            connection.execute("""INSERT INTO audit_events
                (event_id,organization_id,user_id,key_id,action,object_type,object_id,result,reason_code,http_status,request_id,details_json,previous_hash,event_hash,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                event_id, payload["organization_id"], payload["user_id"], payload["key_id"], payload["action"],
                payload["object_type"], payload["object_id"], payload["result"], payload["reason_code"],
                payload["http_status"], payload["request_id"], _canonical(redacted), previous_hash, event_hash, created_at,
            ))
        return event_id

    def list_audit(self, context: SecurityContext, limit: int = 100) -> list[dict[str, Any]]:
        context.require("security:read")
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM audit_events WHERE organization_id=? ORDER BY sequence DESC LIMIT ?", (context.organization_id, min(max(limit, 1), 1000))).fetchall()
        return [dict(row) for row in rows]

    def verify_audit_chain(self) -> dict[str, Any]:
        if not self.audit_hmac_key:
            raise RuntimeError("AUDIT_HMAC_KEY is not configured.")
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM audit_events ORDER BY sequence").fetchall()
        previous = "GENESIS"
        for row in rows:
            payload = {
                "event_id": row["event_id"], "organization_id": row["organization_id"], "user_id": row["user_id"],
                "key_id": row["key_id"], "action": row["action"], "object_type": row["object_type"],
                "object_id": row["object_id"], "result": row["result"], "reason_code": row["reason_code"],
                "http_status": row["http_status"], "request_id": row["request_id"],
                "details": json.loads(row["details_json"]), "previous_hash": row["previous_hash"], "created_at": row["created_at"],
            }
            expected = hmac.new(self.audit_hmac_key, _canonical(payload).encode(), hashlib.sha256).hexdigest()
            if row["previous_hash"] != previous or not hmac.compare_digest(expected, row["event_hash"]):
                return {"valid": False, "events": len(rows), "failed_event_id": row["event_id"]}
            previous = row["event_hash"]
        return {"valid": True, "events": len(rows), "head": previous}

    def create_quarantine_case(self, organization_id: str, document_name: str, findings: list[dict[str, Any]], created_by: str, staged_path: str = "", job_id: str = "") -> str:
        case_id = "qtn_" + uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute("INSERT INTO quarantine_cases VALUES (?,?,?,?, 'quarantined','open',?,?,?,?,NULL,NULL,NULL)",
                               (case_id, organization_id, job_id, _clean(document_name, 240), _canonical(findings), _clean(staged_path, 500), created_by, _now()))
        return case_id

    def list_quarantine(self, context: SecurityContext, status: str | None = None) -> list[dict[str, Any]]:
        if context.role not in {"owner", "admin"}:
            raise AuthorizationError("Quarantine access is restricted.")
        sql, params = "SELECT * FROM quarantine_cases WHERE organization_id=?", [context.organization_id]
        if status:
            sql += " AND status=?"; params.append(status)
        with self._connect() as connection:
            rows = connection.execute(sql + " ORDER BY created_at DESC", params).fetchall()
        result = []
        for row in rows:
            value = dict(row); value["findings"] = json.loads(value.pop("findings_json")); value.pop("staged_path", None); result.append(value)
        return result

    def get_quarantine_internal(self, context: SecurityContext, case_id: str) -> dict[str, Any] | None:
        if context.role not in {"owner", "admin"}:
            raise AuthorizationError("Quarantine access is restricted.")
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM quarantine_cases WHERE case_id=? AND organization_id=?", (case_id, context.organization_id)).fetchone()
        if not row:
            return None
        value = dict(row)
        value["findings"] = json.loads(value.pop("findings_json"))
        return value

    def resolve_quarantine(self, context: SecurityContext, case_id: str, decision: str, reason: str) -> None:
        if context.role not in {"owner", "admin"} or decision not in {"approved", "rejected"} or not reason.strip():
            raise AuthorizationError("A reasoned owner/admin quarantine decision is required.")
        with self._connect() as connection:
            cursor = connection.execute("UPDATE quarantine_cases SET status=?,resolved_by=?,resolution_reason=?,resolved_at=? WHERE case_id=? AND organization_id=? AND status='open'", (decision, context.user_id, _clean(reason, 1000), _now(), case_id, context.organization_id))
            if not cursor.rowcount:
                raise KeyError("Quarantine case not found.")
        self.append_audit(context.organization_id, context.user_id, context.key_id, f"quarantine.{decision}", "quarantine_case", case_id, "success", 200, details={"reason_code": "HUMAN_REVIEW"})
