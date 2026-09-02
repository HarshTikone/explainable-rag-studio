import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
from backend.security_models import ROLE_SCOPES, SecurityContext
from backend.tenant_store import TenantStoreManager
from backend.security import SecurityRegistry


def context(org, user="user", role="owner", scopes=None):
    return SecurityContext(org, user, role, "key", frozenset(scopes or ROLE_SCOPES[role]))


def test_object_ids_are_tenant_scoped_and_return_generic_404(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "tenant_stores", TenantStoreManager(str(tmp_path / "organizations")))
    api._runtimes.clear()
    org_a, org_b = context("org_a", "alice"), context("org_b", "bob")
    registry_a = api._runtime(org_a)[0]
    job_id = registry_a.enqueue_job({"organization_id": "org_a"}, created_by="alice")
    assert api.get_job(job_id, org_a)["job_id"] == job_id
    with pytest.raises(HTTPException) as error:
        api.get_job(job_id, org_b)
    assert error.value.status_code == 404
    case_id = api._runtime(org_a)[3].enqueue({"text": "claim", "evidence": []}, "low_confidence", "config")
    assert api.review(case_id, org_a)["case_id"] == case_id
    with pytest.raises(HTTPException) as error:
        api.review(case_id, org_b)
    assert error.value.status_code == 404


def test_role_and_key_scopes_reduce_server_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "tenant_stores", TenantStoreManager(str(tmp_path / "organizations")))
    api._runtimes.clear()
    viewer = context("org_a", "viewer", "viewer")
    assert api.documents(viewer) == {"documents": []}
    with pytest.raises(HTTPException) as error:
        api.reviews(context=viewer)
    assert error.value.status_code == 403


def test_http_auth_contract_and_sensitive_key_absence(tmp_path, monkeypatch):
    registry = SecurityRegistry(str(tmp_path / "security.db"), "pepper", "audit")
    boot = registry.bootstrap("Example", "owner@example.test")
    monkeypatch.setattr(api, "security_registry", registry)
    monkeypatch.setattr(api, "tenant_stores", TenantStoreManager(str(tmp_path / "organizations")))
    api._runtimes.clear()
    client = TestClient(api.app)
    assert client.get("/auth/me").status_code == 401
    headers = {"Authorization": f"Bearer {boot['api_key']}"}
    identity = client.get("/auth/me", headers=headers)
    assert identity.status_code == 200 and identity.json()["organization_id"] == boot["organization_id"]
    listed = client.get("/api-keys", headers=headers)
    assert listed.status_code == 200
    assert boot["api_key"] not in listed.text and "digest" not in listed.text
    assert client.get("/reviews", headers=headers).status_code == 200
    assert client.get("/health").status_code == 200
    reduced_owner = context("org_a", "owner", "owner", {"query"})
    with pytest.raises(HTTPException) as error:
        api.documents(reduced_owner)
    assert error.value.status_code == 403
