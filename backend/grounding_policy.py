"""Versioned, fingerprinted claim-verification policy contracts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .config import SETTINGS


GROUNDING_POLICY_SCHEMA_VERSION = "1.0"
DEFAULT_PREMISE_VERSION = "context-envelope-v1"
DEFAULT_GUARD_VERSION = "deterministic-guards-v1"


@dataclass(frozen=True)
class GroundingPolicy:
    model_name: str = SETTINGS.grounding_model
    model_revision: str = SETTINGS.grounding_model_revision
    entailment_threshold: float = SETTINGS.grounding_entailment_threshold
    contradiction_threshold: float = SETTINGS.grounding_contradiction_threshold
    low_confidence_margin: float = SETTINGS.grounding_low_confidence_margin
    conflict_relevance_threshold: float = SETTINGS.grounding_conflict_relevance_threshold
    premise_strategy: str = SETTINGS.grounding_premise_strategy
    premise_version: str = DEFAULT_PREMISE_VERSION
    guard_version: str = DEFAULT_GUARD_VERSION
    evidence_scan_k: int = SETTINGS.grounding_evidence_scan_k
    batch_size: int = SETTINGS.grounding_batch_size
    max_length: int = SETTINGS.grounding_max_length
    schema_version: str = GROUNDING_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.premise_strategy not in {"atomic_sentence", "context_envelope"}:
            raise ValueError("premise_strategy must be atomic_sentence or context_envelope.")
        for name, value in (
            ("entailment_threshold", self.entailment_threshold),
            ("contradiction_threshold", self.contradiction_threshold),
            ("low_confidence_margin", self.low_confidence_margin),
            ("conflict_relevance_threshold", self.conflict_relevance_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.evidence_scan_k < 1 or self.batch_size < 1 or self.max_length < 32:
            raise ValueError("Policy limits must be positive and max_length must be at least 32.")

    @property
    def policy_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "gpol_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "policy_id": self.policy_id}

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "GroundingPolicy":
        fields = {key: item for key, item in value.items() if key != "policy_id"}
        policy = cls(**fields)
        supplied_id = value.get("policy_id")
        if supplied_id and supplied_id != policy.policy_id:
            raise ValueError("Grounding policy fingerprint does not match its contents.")
        return policy


def default_grounding_policy() -> GroundingPolicy:
    return GroundingPolicy()
