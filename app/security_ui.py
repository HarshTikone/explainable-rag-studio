"""Shared Streamlit authentication and tenant routing."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from backend.config import SETTINGS
from backend.security import SecurityRegistry
from backend.security_models import AuthenticationError, AuthorizationError, ROLE_SCOPES, SecurityContext
from backend.tenant_store import TenantStoreManager
from backend.vectorstore import FaissStore
from app.api_client import ApiClientError, RagApiClient


@st.cache_resource
def security_registry() -> SecurityRegistry:
    value = SecurityRegistry(SETTINGS.security_db_path, SETTINGS.api_key_pepper, SETTINGS.audit_hmac_key)
    value.ensure_public_organization(SETTINGS.public_organization_id)
    return value


@st.cache_resource
def tenant_manager() -> TenantStoreManager:
    return TenantStoreManager(SETTINGS.tenant_index_root, SETTINGS.tenant_store_cache_size)


def security_context(scope: str | None = None, *, allow_anonymous: bool = False) -> SecurityContext:
    with st.sidebar:
        st.markdown("### Workspace access")
        raw_key = st.text_input("API key", type="password", key="security_api_key", help="Kept only in this browser session.")
        if raw_key and st.button("Clear key", use_container_width=True):
            st.session_state.pop("security_api_key", None)
            st.rerun()
    if SETTINGS.platform_mode == "postgres":
        session_cookie = ""
        try:
            session_cookie = st.context.cookies.get("rag_session", "")
        except Exception:
            pass
        if not raw_key and not session_cookie:
            with st.sidebar:
                st.link_button("Sign in with SSO", f"{SETTINGS.api_base_url}/auth/login", use_container_width=True)
            if SETTINGS.security_mode == "demo" and allow_anonymous:
                return SecurityContext(SETTINGS.public_organization_id, "anonymous_demo", "viewer", None, ROLE_SCOPES["viewer"], True)
            st.info("Sign in with SSO or enter a scoped API key.")
            st.stop()
        try:
            identity = RagApiClient(raw_key, session_cookie=session_cookie).me()
            context = SecurityContext(identity["organization_id"], identity["user_id"], identity["role"], identity.get("key_id"), frozenset(identity["scopes"]))
        except ApiClientError as exc:
            st.error(str(exc))
            st.stop()
    elif raw_key:
        try:
            context = security_registry().authenticate(raw_key)
        except AuthenticationError:
            st.error("That API key is invalid, expired, or revoked.")
            st.stop()
    elif SETTINGS.security_mode == "demo" and allow_anonymous:
        context = SecurityContext(SETTINGS.public_organization_id, "anonymous_demo", "viewer", None, ROLE_SCOPES["viewer"], True)
    else:
        if SETTINGS.security_mode == "required" and not security_registry().protected_ready:
            st.error("Protected mode needs API_KEY_PEPPER and AUDIT_HMAC_KEY before it can start.")
        else:
            st.info("Enter an API key to open this workspace.")
        st.stop()
    if context.anonymous_demo and not allow_anonymous:
        st.info("This page requires an authenticated organization API key.")
        st.stop()
    if scope:
        try:
            context.require(scope)
        except AuthorizationError:
            st.error("Your role or API-key scope does not allow this page.")
            st.stop()
    with st.sidebar:
        st.caption(f"{context.role.title()} · {context.organization_id}")
    return context


def tenant_store(context: SecurityContext) -> FaissStore:
    if SETTINGS.platform_mode == "postgres":
        raise RuntimeError("Production Streamlit pages must use RagApiClient instead of direct storage access.")
    value = tenant_manager().get(context.organization_id, reload=True)
    if value.index is None and context.organization_id == SETTINGS.public_organization_id:
        value = FaissStore(SETTINGS.index_dir)
        value.load()
    return value


def tenant_dir(context: SecurityContext) -> Path:
    return tenant_manager().organization_dir(context.organization_id)
