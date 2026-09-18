"""Typed contracts for structured claim generation and evidence verification."""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


ClaimVerdict = Literal["supported", "unsupported", "contradicted", "disputed", "invalid_citation"]
GroundingStatus = Literal["verified", "partial", "abstained", "verification_unavailable"]


class DraftClaim(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    cited_chunk_ids: List[str] = Field(min_length=1, max_length=3)
    provenance: Literal["generated", "extractive"] = "generated"

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("Claim text cannot be empty.")
        return normalized

    @field_validator("cited_chunk_ids")
    @classmethod
    def unique_citations(cls, value: List[str]) -> List[str]:
        normalized = list(dict.fromkeys(chunk_id.strip() for chunk_id in value if chunk_id.strip()))
        if not normalized:
            raise ValueError("Every claim requires at least one citation.")
        return normalized


class StructuredDraft(BaseModel):
    answerable: bool
    claims: List[DraftClaim] = Field(default_factory=list, max_length=8)


class EvidenceSelection(BaseModel):
    """A provider-selected sentence address, resolved to exact text locally."""

    chunk_id: str = Field(min_length=1, max_length=120)
    sentence_index: int = Field(ge=0)


class PublicEvidenceDraft(BaseModel):
    """Bounded public-provider response with no generated answer text."""

    answerable: bool
    selections: List[EvidenceSelection] = Field(default_factory=list, max_length=2)


class ClaimEvidence(BaseModel):
    chunk_id: str
    source: str = ""
    page: int | str | None = None
    page_range: List[int] = Field(default_factory=list)
    heading_path: List[str] = Field(default_factory=list)
    document_id: str = ""
    document_version_id: str = ""
    excerpt: str = ""
    cited: bool = False
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0
    relevance_score: float = 0.0
    premise_version: str = ""
    candidate_premises: List[str] = Field(default_factory=list)
    raw_scores: List[Dict[str, float]] = Field(default_factory=list)
    guard_reasons: List[str] = Field(default_factory=list)
    semantic_similarity: float = 0.0
    anchor_match: bool = False


class ClaimVerification(BaseModel):
    claim_id: str
    text: str
    cited_chunk_ids: List[str]
    provenance: Literal["generated", "extractive"]
    verdict: ClaimVerdict
    accepted: bool
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    low_confidence: bool = False
    reason_codes: List[str] = Field(default_factory=list)
    evidence: List[ClaimEvidence] = Field(default_factory=list)
    decision_path: List[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    schema_version: str = "1.1"
    status: GroundingStatus
    has_conflicts: bool = False
    accepted_claims: List[ClaimVerification] = Field(default_factory=list)
    rejected_claims: List[ClaimVerification] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    verifier: Dict[str, Any] = Field(default_factory=dict)
    policy_id: str = ""
    premise_version: str = ""
    guard_version: str = ""
    latency_ms: Dict[str, float] = Field(default_factory=dict)
