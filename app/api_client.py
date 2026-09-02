"""The production Streamlit frontend communicates only through FastAPI."""
from __future__ import annotations

from typing import Any

import httpx

from backend.config import SETTINGS


class ApiClientError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class RagApiClient:
    def __init__(self, bearer_token: str = "", base_url: str = SETTINGS.api_base_url, session_cookie: str = ""):
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token
        self.session_cookie = session_cookie

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.bearer_token}"} if self.bearer_token else {}

    def request(self, method: str, path: str, **kwargs) -> Any:
        headers = {**self.headers, **kwargs.pop("headers", {})}
        try:
            cookies = {"rag_session": self.session_cookie} if self.session_cookie else None
            response = httpx.request(method, f"{self.base_url}{path}", headers=headers, cookies=cookies, timeout=60.0, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiClientError(503, "The API is unavailable.") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise ApiClientError(response.status_code, str(detail))
        if response.status_code == 204:
            return None
        return response.json()

    def me(self):
        return self.request("GET", "/auth/me")

    def ask(self, question: str, top_k: int, strategy: str):
        return self.request("POST", "/ask", json={"question": question, "top_k": top_k, "retrieval_strategy": strategy})

    def upload(self, files, options: dict[str, Any]):
        multipart = [("files", (name, content, mime)) for name, content, mime in files]
        return self.request("POST", "/ingestion/jobs", files=multipart, data=options)

    def documents(self):
        return self.request("GET", "/documents")

    def job(self, job_id: str):
        return self.request("GET", f"/ingestion/jobs/{job_id}")

    def retry_job(self, job_id: str):
        return self.request("POST", f"/ingestion/jobs/{job_id}/retry")

    def cancel_job(self, job_id: str):
        return self.request("POST", f"/ingestion/jobs/{job_id}/cancel")

    def delete_document(self, document_id: str):
        return self.request("DELETE", f"/documents/{document_id}")

    def reviews(self, status: str):
        return self.request("GET", "/reviews", params={"status": status, "limit": 200})

    def review(self, case_id: str):
        return self.request("GET", f"/reviews/{case_id}")

    def decide(self, case_id: str, decision: str, notes: str):
        return self.request("POST", f"/reviews/{case_id}/decision", json={"decision": decision, "reviewer": "oidc", "notes": notes})

    def keys(self):
        return self.request("GET", "/api-keys")

    def members(self):
        return self.request("GET", "/members")

    def quarantine(self):
        return self.request("GET", "/quarantine")

    def audit(self):
        return self.request("GET", "/audit", params={"limit": 1000})

    def verify_audit(self):
        return self.request("GET", "/audit/verify")
