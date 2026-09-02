import pytest

from backend.grounding_policy import GroundingPolicy


def test_policy_fingerprint_round_trip_and_tamper_detection():
    policy = GroundingPolicy(entailment_threshold=0.65, conflict_relevance_threshold=0.4)
    payload = policy.to_dict()
    assert GroundingPolicy.from_dict(payload) == policy
    payload["entailment_threshold"] = 0.7
    with pytest.raises(ValueError, match="fingerprint"):
        GroundingPolicy.from_dict(payload)


def test_policy_rejects_invalid_limits_and_strategy():
    with pytest.raises(ValueError):
        GroundingPolicy(entailment_threshold=1.1)
    with pytest.raises(ValueError):
        GroundingPolicy(premise_strategy="unknown")
