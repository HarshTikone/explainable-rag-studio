from typing import Literal

from pydantic import BaseModel, Field

from .config import SETTINGS


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=SETTINGS.top_k, ge=1, le=50)
    retrieval_strategy: Literal["lexical", "dense", "dense_mmr", "hybrid_rrf", "hybrid_rerank"] = "dense_mmr"
    rerank_candidates: int | None = Field(default=None, ge=1, le=100)


class ReviewDecisionRequest(BaseModel):
    decision: Literal["supported", "unsupported", "contradicted"]
    reviewer: str = Field(default="local", min_length=1, max_length=100)
    notes: str = Field(default="", max_length=2000)


class ApiKeyCreateRequest(BaseModel):
    scopes: list[Literal["query", "documents:read", "documents:write", "reviews:write", "experiments:run", "security:read", "members:write"]]
    expires_days: int = Field(default=90, ge=1, le=365)


class MembershipCreateRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    display_name: str = Field(min_length=1, max_length=120)
    role: Literal["owner", "admin", "editor", "viewer"]


class MembershipUpdateRequest(BaseModel):
    role: Literal["owner", "admin", "editor", "viewer"] | None = None


class OidcIdentityRequest(BaseModel):
    issuer: str = Field(min_length=8, max_length=500)
    subject: str = Field(min_length=1, max_length=500)


class QuarantineDecisionRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=1000)
