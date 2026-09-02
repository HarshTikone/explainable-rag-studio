import pytest
from pydantic import ValidationError

from backend.schemas import AskRequest


def test_api_request_defaults_to_dense_mmr():
    assert AskRequest(question="hello").retrieval_strategy == "dense_mmr"
    assert AskRequest(question="hello", retrieval_strategy="hybrid_rerank").retrieval_strategy == "hybrid_rerank"


def test_api_request_rejects_invalid_strategy_and_bounds():
    with pytest.raises(ValidationError):
        AskRequest(question="hello", retrieval_strategy="invalid")
    with pytest.raises(ValidationError):
        AskRequest(question="", top_k=100)
    with pytest.raises(ValidationError):
        AskRequest(question="hello", retrieval_strategy="hybrid_rerank", rerank_candidates=0)
    with pytest.raises(ValidationError):
        AskRequest(question="hello", retrieval_strategy="hybrid_rerank", rerank_candidates=101)
