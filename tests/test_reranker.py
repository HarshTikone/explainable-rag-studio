import numpy as np
import pytest

from backend.reranker import CrossEncoderReranker, RerankerUnavailableError, get_default_reranker


class FakeModel:
    def __init__(self):
        self.call = None

    def predict(self, pairs, **kwargs):
        self.call = (pairs, kwargs)
        return np.asarray([len(document) for _, document in pairs], dtype="float32")


class BrokenModel:
    def predict(self, pairs, **kwargs):
        raise RuntimeError("broken")


def test_cross_encoder_batches_pairs_and_propagates_scores():
    reranker = CrossEncoderReranker("fake", batch_size=2)
    reranker._model = FakeModel()
    assert reranker.score("query", ["a", "longer"]) == [1.0, 6.0]
    pairs, options = reranker._model.call
    assert pairs == [("query", "a"), ("query", "longer")]
    assert options["batch_size"] == 2


def test_reranker_handles_empty_and_model_failure():
    reranker = CrossEncoderReranker("fake")
    assert reranker.score("query", []) == []
    reranker._model = BrokenModel()
    with pytest.raises(RerankerUnavailableError):
        reranker.score("query", ["document"])


def test_reranker_wraps_model_load_failure(monkeypatch):
    reranker = CrossEncoderReranker("missing-model")

    def fail_load():
        raise RerankerUnavailableError("model unavailable")

    monkeypatch.setattr(reranker, "_load", fail_load)
    with pytest.raises(RerankerUnavailableError, match="model unavailable"):
        reranker.score("query", ["document"])


def test_default_reranker_is_cached():
    assert get_default_reranker("fake-model") is get_default_reranker("fake-model")
