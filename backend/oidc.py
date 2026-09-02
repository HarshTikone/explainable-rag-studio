"""Generic OIDC validation and PKCE state storage.

OIDC identity establishes a user subject only. Organization membership and
effective permissions are always resolved from the application registry.
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
import jwt

from .security_models import AuthenticationError, SecurityContext


@dataclass(frozen=True)
class OidcPrincipal:
    subject: str
    issuer: str
    email: str
    display_name: str
    claims: dict


class OidcValidator:
    def __init__(self, issuer: str, audience: str, jwks_ttl_seconds: int = 300, http_client=None):
        self.issuer = issuer.rstrip("/")
        self.audience = audience
        self.jwks_ttl_seconds = jwks_ttl_seconds
        self.http = http_client or httpx.Client(timeout=5.0)
        self._configuration: dict | None = None
        self._jwks: dict[str, dict] = {}
        self._expires_at = 0.0

    def _refresh(self) -> None:
        try:
            configuration = self.http.get(f"{self.issuer}/.well-known/openid-configuration").json()
            if configuration.get("issuer", "").rstrip("/") != self.issuer:
                raise AuthenticationError("OIDC issuer metadata mismatch.")
            jwks = self.http.get(configuration["jwks_uri"]).json()
            self._configuration = configuration
            self._jwks = {key["kid"]: key for key in jwks.get("keys", []) if key.get("kid")}
            self._expires_at = time.monotonic() + self.jwks_ttl_seconds
        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationError("OIDC discovery is unavailable.") from exc

    def validate(self, token: str, nonce: str | None = None) -> OidcPrincipal:
        if not token:
            raise AuthenticationError("Missing OIDC token.")
        try:
            header = jwt.get_unverified_header(token)
            if time.monotonic() >= self._expires_at or header.get("kid") not in self._jwks:
                self._refresh()
            key = jwt.PyJWK.from_dict(self._jwks[header["kid"]]).key
            claims = jwt.decode(
                token,
                key,
                algorithms=[header.get("alg", "RS256")],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat", "iss", "sub"]},
            )
            if nonce is not None and not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
                raise AuthenticationError("OIDC nonce mismatch.")
        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationError("Invalid OIDC token.") from exc
        return OidcPrincipal(
            subject=str(claims["sub"]),
            issuer=str(claims["iss"]),
            email=str(claims.get("email", "")),
            display_name=str(claims.get("name", claims.get("preferred_username", "User"))),
            claims=claims,
        )

    @property
    def authorization_endpoint(self) -> str:
        if not self._configuration or time.monotonic() >= self._expires_at:
            self._refresh()
        return str(self._configuration["authorization_endpoint"])

    @property
    def token_endpoint(self) -> str:
        if not self._configuration or time.monotonic() >= self._expires_at:
            self._refresh()
        return str(self._configuration["token_endpoint"])


class PkceStateStore:
    def __init__(self, redis_client, ttl_seconds: int = 300):
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _challenge(verifier: str) -> str:
        return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()

    def begin(self, validator: OidcValidator, client_id: str, redirect_uri: str) -> dict[str, str]:
        state, nonce, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(64)
        payload = json.dumps({"nonce": nonce, "verifier": verifier, "redirect_uri": redirect_uri})
        self.redis.setex(f"rag:oidc:{state}", self.ttl_seconds, payload)
        query = urlencode({
            "response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
            "scope": "openid email profile", "state": state, "nonce": nonce,
            "code_challenge": self._challenge(verifier), "code_challenge_method": "S256",
        })
        return {"state": state, "authorization_url": f"{validator.authorization_endpoint}?{query}"}

    def consume(self, state: str) -> dict[str, str]:
        key = f"rag:oidc:{state}"
        pipe = self.redis.pipeline()
        pipe.get(key)
        pipe.delete(key)
        value, _ = pipe.execute()
        if not value:
            raise AuthenticationError("OIDC state is invalid or expired.")
        return json.loads(value)
