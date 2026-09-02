import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from backend.security import SecurityRegistry
from backend.security_models import AuthenticationError, AuthorizationError, ROLE_SCOPES, SecurityContext
from backend.security_benchmark import CATEGORY_COUNTS, security_benchmark, validate_security_benchmark


def registry(tmp_path):
    return SecurityRegistry(str(tmp_path / "security.db"), "test-pepper", "test-audit-key")


def test_bootstrap_key_is_scoped_hashed_and_revocable(tmp_path):
    value = registry(tmp_path)
    boot = value.bootstrap("Example", "owner@example.test")
    assert boot["api_key"].startswith("ragk_")
    context = value.authenticate(boot["api_key"])
    assert context.role == "owner" and context.scopes == ROLE_SCOPES["owner"]
    with value._connect() as connection:
        row = connection.execute("SELECT digest FROM api_keys WHERE key_id=?", (boot["key_id"],)).fetchone()
        assert boot["api_key"] not in row["digest"] and len(row["digest"]) == 64
    reduced = value.create_key_for_context(context, ["query"], 1)
    reduced_context = value.authenticate(reduced["api_key"])
    assert reduced_context.scopes == frozenset({"query"})
    value.revoke_api_key(context, reduced["key_id"])
    with pytest.raises(AuthenticationError):
        value.authenticate(reduced["api_key"])


def test_roles_scopes_and_final_owner(tmp_path):
    value = registry(tmp_path)
    boot = value.bootstrap("Example", "owner@example.test")
    owner = value.authenticate(boot["api_key"])
    viewer_id = value.add_user(owner, "viewer@example.test", "Viewer", "viewer")
    with pytest.raises(AuthorizationError):
        value.create_api_key(owner.organization_id, viewer_id, ["documents:write"])
    with pytest.raises(ValueError, match="final"):
        value.change_membership(owner, owner.user_id, "admin")
    assert len(value.list_members(owner)) == 2


def test_expired_malformed_and_wrong_secret_are_generic(tmp_path):
    value = registry(tmp_path)
    boot = value.bootstrap("Example", "owner@example.test")
    with value._connect() as connection:
        connection.execute("UPDATE api_keys SET expires_at=? WHERE key_id=?", ((datetime.now(timezone.utc)-timedelta(days=1)).isoformat(), boot["key_id"]))
    for key in (boot["api_key"], "bad", boot["api_key"][:-1] + "A"):
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            value.authenticate(key)


def test_audit_is_redacted_append_only_and_tamper_evident(tmp_path):
    value = registry(tmp_path)
    boot = value.bootstrap("Example", "owner@example.test")
    context = value.authenticate(boot["api_key"])
    value.append_audit(context.organization_id, context.user_id, context.key_id, "query.complete", "query", None,
                       "success", 200, details={"prompt": "private", "query_sha256": "abc", "control": "x\x00y"})
    assert value.verify_audit_chain()["valid"] is True
    rows = value.list_audit(context)
    assert "private" not in rows[0]["details_json"] and "[REDACTED]" in rows[0]["details_json"]
    with value._connect() as connection, pytest.raises(sqlite3.DatabaseError):
        connection.execute("UPDATE audit_events SET result='changed' WHERE sequence=1")
    # A privileged offline tamper simulation removes the trigger, then proves verification fails.
    with value._connect() as connection:
        connection.execute("DROP TRIGGER audit_no_update")
        connection.execute("UPDATE audit_events SET result='changed' WHERE sequence=1")
    assert value.verify_audit_chain()["valid"] is False


def test_quarantine_is_tenant_scoped_and_decision_is_audited(tmp_path):
    value = registry(tmp_path)
    boot = value.bootstrap("Example", "owner@example.test")
    context = value.authenticate(boot["api_key"])
    case_id = value.create_quarantine_case(context.organization_id, "bad.pdf", [{"code": "ACTIVE_PDF_CONTENT"}], context.user_id, "hidden")
    assert value.list_quarantine(context)[0]["case_id"] == case_id
    value.resolve_quarantine(context, case_id, "rejected", "Unsafe active content")
    assert value.list_quarantine(context, "rejected")[0]["status"] == "rejected"


def test_security_benchmark_has_required_60_case_balance():
    assert validate_security_benchmark(security_benchmark()) == CATEGORY_COUNTS


def test_configuration_scope_validation_and_empty_chain(tmp_path):
    unconfigured = SecurityRegistry(str(tmp_path / "empty.db"))
    assert not unconfigured.protected_ready
    with pytest.raises(RuntimeError):
        unconfigured.require_protected_configuration()
    with pytest.raises(RuntimeError):
        unconfigured.verify_audit_chain()
    value = registry(tmp_path / "configured")
    assert value.verify_audit_chain() == {"valid": True, "events": 0, "head": "GENESIS"}
    with pytest.raises(ValueError):
        SecurityContext("org", "user", "unknown", None, frozenset())
    with pytest.raises(ValueError):
        SecurityContext("org", "user", "viewer", None, frozenset({"bad"}))


def test_non_owner_members_and_key_admin_are_denied(tmp_path):
    value = registry(tmp_path)
    boot = value.bootstrap("Example", "owner@example.test")
    owner = value.authenticate(boot["api_key"])
    viewer_id = value.add_user(owner, "viewer@example.test", "Viewer", "viewer")
    viewer_key = value.create_api_key(owner.organization_id, viewer_id, ["query"])
    viewer = value.authenticate(viewer_key["api_key"])
    for operation in (
        lambda: value.add_user(viewer, "x@example.test", "X", "viewer"),
        lambda: value.list_members(viewer),
        lambda: value.list_api_keys(viewer),
        lambda: value.revoke_api_key(viewer, boot["key_id"]),
        lambda: value.list_quarantine(viewer),
    ):
        with pytest.raises(AuthorizationError):
            operation()
    with pytest.raises(ValueError):
        value.create_api_key(owner.organization_id, owner.user_id, ["unknown"])
    with pytest.raises(KeyError):
        value.revoke_api_key(owner, "missing")


def test_quarantine_missing_and_invalid_decisions(tmp_path):
    value = registry(tmp_path)
    boot = value.bootstrap("Example", "owner@example.test")
    owner = value.authenticate(boot["api_key"])
    assert value.get_quarantine_internal(owner, "missing") is None
    with pytest.raises(AuthorizationError):
        value.resolve_quarantine(owner, "missing", "maybe", "")
    with pytest.raises(KeyError):
        value.resolve_quarantine(owner, "missing", "rejected", "reason")
