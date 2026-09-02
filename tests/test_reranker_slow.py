import os

import pytest

from backend.config import SETTINGS
from backend.reranker import CrossEncoderReranker
from backend.grounding import CrossEncoderNliVerifier


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("RUN_RERANKER_SMOKE") != "1", reason="release validation only")
def test_real_minilm_reranker_smoke():
    reranker = CrossEncoderReranker(
        SETTINGS.reranker_model, batch_size=2, backend="onnx",
        model_revision=SETTINGS.reranker_model_revision, onnx_file=SETTINGS.reranker_onnx_file,
    )
    scores = reranker.score(
        "Which procedure rotates signing keys?",
        ["The current procedure rotates signing keys every 30 days.", "Cafeteria hours begin at noon."],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("RUN_GROUNDING_SMOKE") != "1", reason="release validation only")
def test_real_deberta_grounding_smoke():
    verifier = CrossEncoderNliVerifier(
        SETTINGS.grounding_model, SETTINGS.grounding_model_revision, batch_size=2,
        backend="onnx", onnx_file=SETTINGS.grounding_onnx_file,
    )
    scores = verifier.score([
        ("Meridian tokens last 45 minutes.", "Meridian tokens expire after 45 minutes."),
        ("Meridian tokens last 45 minutes.", "Meridian tokens last eight hours."),
    ])
    assert len(scores) == 2
    assert scores[0]["entailment"] > scores[0]["contradiction"]
    assert scores[1]["contradiction"] > scores[1]["entailment"]
