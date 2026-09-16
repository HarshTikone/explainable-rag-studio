import json
from types import SimpleNamespace

import numpy as np
import pytest

import backend.grounding as grounding_module
from backend.grounding import (
    CrossEncoderNliVerifier,
    VerifierUnavailableError,
    build_extractive_draft,
    citations_from_grounding,
    deterministic_guards,
    generate_structured_draft,
    render_grounded_answer,
    select_evidence_excerpt,
    select_evidence_premise,
    verify_claims,
)
from backend.grounding_models import DraftClaim, StructuredDraft
from backend.grounding_policy import GroundingPolicy
from backend.qa import answer_with_optional_llm


class FakeVerifier:
    model_name = "fake-nli"
    model_revision = "test"

    def score(self, pairs):
        rows = []
        for premise, claim in pairs:
            key = (premise + " " + claim).casefold()
            if "opposite" in premise.casefold():
                rows.append({"contradiction": 0.92, "entailment": 0.03, "neutral": 0.05})
            elif "unknown assertion" in claim.casefold():
                rows.append({"contradiction": 0.04, "entailment": 0.10, "neutral": 0.86})
            else:
                rows.append({"contradiction": 0.02, "entailment": 0.95, "neutral": 0.03})
        return rows


class BrokenVerifier(FakeVerifier):
    def score(self, pairs):
        raise VerifierUnavailableError("offline")


def hit(chunk_id, text):
    return (1.0, {
        "chunk_id": chunk_id, "text": text, "generation_text": text,
        "source": f"{chunk_id}.md", "page": 1, "heading_path": ["Test"],
        "document_id": "doc", "document_version_id": "ver",
    })


def test_supported_claim_is_rendered_and_citations_are_claim_bound():
    retrieved = [hit("c1", "Meridian tokens last 45 minutes.")]
    draft = StructuredDraft(answerable=True, claims=[DraftClaim(
        text="Meridian tokens last 45 minutes.", cited_chunk_ids=["c1"]
    )])
    result = verify_claims(draft, retrieved, FakeVerifier()).result
    assert result.status == "verified"
    assert result.accepted_claims[0].verdict == "supported"
    assert render_grounded_answer(result).endswith("[c1]")
    citations = citations_from_grounding(result)
    assert citations[0]["claim_ids"] == ["clm_001"]
    assert citations[0]["document_version_id"] == "ver"


def test_invalid_unsupported_and_conflicting_claims_are_never_rendered():
    retrieved = [
        hit("c1", "The policy says the feature is enabled."),
        hit("c2", "The opposite feature policy says the feature is disabled."),
    ]
    draft = StructuredDraft(answerable=True, claims=[
        DraftClaim(text="The feature is enabled.", cited_chunk_ids=["c1"]),
        DraftClaim(text="Unknown assertion.", cited_chunk_ids=["c1"]),
        DraftClaim(text="A fabricated claim.", cited_chunk_ids=["missing"]),
    ])
    result = verify_claims(draft, retrieved, FakeVerifier()).result
    assert {claim.verdict for claim in result.rejected_claims} == {"disputed", "unsupported", "invalid_citation"}
    assert render_grounded_answer(result) == "I don't know based on the retrieved evidence."
    assert citations_from_grounding(result) == []
    assert result.has_conflicts


def test_deterministic_guards_check_numbers_identifiers_and_negation():
    reasons = deterministic_guards("AU-4012 does not expire after 45 minutes.", "AU-4012 expires after 30 minutes.")
    assert "MISSING_IDENTIFIER_OR_NUMBER" in reasons
    assert "NEGATION_NOT_IN_EVIDENCE" in reasons
    assert "CONFLICTING_IDENTIFIER_OR_NUMBER" in reasons
    assert "NEGATION_CONFLICT" in reasons


def test_claim_relevant_evidence_sentence_is_selected_for_nli_and_display():
    text = "The service uses dense retrieval. RRF combines dense and BM25 with constant 60. MMR is not used after fusion."
    assert select_evidence_excerpt(text, "The RRF constant is 60.") == "RRF combines dense and BM25 with constant 60."
    item = {
        "generation_text": text, "title": "Retrieval Guide", "heading_path": ["Hybrid", "RRF"]
    }
    premise = select_evidence_premise(item, "The RRF constant is 60.", GroundingPolicy())
    assert "Document: Retrieval Guide" in premise
    assert "Section: Hybrid > RRF" in premise
    assert "RRF combines dense and BM25 with constant 60." in premise


def test_verifier_failure_only_allows_exact_extractive_claims():
    retrieved = [hit("c1", "This exact sentence is evidence.")]
    extractive = build_extractive_draft(retrieved)
    extractive_result = verify_claims(extractive, retrieved, BrokenVerifier()).result
    assert extractive_result.status == "verified"

    generated = StructuredDraft(answerable=True, claims=[DraftClaim(
        text="This exact sentence is evidence.", cited_chunk_ids=["c1"], provenance="generated"
    )])
    generated_result = verify_claims(generated, retrieved, BrokenVerifier()).result
    assert generated_result.status == "verification_unavailable"
    assert "VERIFIER_UNAVAILABLE" in generated_result.rejected_claims[0].reason_codes


def test_offline_qa_returns_only_verified_extractive_claims(tmp_path):
    retrieved = [hit("c1", "The current RRF constant is 60. A second supported fact exists.")]
    output = answer_with_optional_llm(
        "What is the constant?", retrieved, False, verifier=FakeVerifier(), persist_review=False
    )
    assert output["grounding"]["status"] == "verified"
    assert "[c1]" in output["answer"]
    assert output["citations"][0]["verification_score"] == 1.0


def test_question_aware_extractive_answer_selects_the_relevant_sentence():
    retrieved = [hit(
        "c1",
        "Aegis Compliance Current Matrix. Audit events are retained for 400 days. Invoice evidence lasts seven years.",
    )]
    output = answer_with_optional_llm(
        "What is the audit log retention period?", retrieved, False,
        verifier=BrokenVerifier(), persist_review=False, extractive_max_claims=1,
    )
    assert "400 days" in output["answer"]
    assert "seven years" not in output["answer"]


class FakeGeminiModels:
    def __init__(self):
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(text="not-json", parsed=None)
        return SimpleNamespace(
            text=json.dumps({"answerable": True, "claims": [{"text": "A supported claim.", "cited_chunk_ids": ["c1"]}]}),
            parsed=None,
        )


def test_structured_generation_retries_once_after_malformed_output():
    models = FakeGeminiModels()
    draft = generate_structured_draft("Question?", [hit("c1", "A supported claim.")], SimpleNamespace(models=models), "fake")
    assert models.calls == 2
    assert draft.claims[0].provenance == "generated"


class FakeCrossEncoderModel:
    def __init__(self, logits=None, error=None, labels=None):
        self.logits = np.asarray(logits if logits is not None else [[1.0, 3.0, 0.0]])
        self.error = error
        self.model = SimpleNamespace(config=SimpleNamespace(
            id2label=labels or {0: "contradiction", 1: "entailment", 2: "neutral"}
        ))

    def predict(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.logits


def test_cross_encoder_wrapper_normalizes_labels_and_failures(monkeypatch):
    verifier = CrossEncoderNliVerifier("fake", model_revision="rev", batch_size=2)
    assert not verifier.is_loaded
    verifier._model = FakeCrossEncoderModel()
    assert verifier.is_loaded
    assert verifier.score([]) == []
    score = verifier.score([("evidence", "claim")])[0]
    assert score["entailment"] > score["contradiction"]
    assert verifier._load() is verifier._model

    verifier._model = FakeCrossEncoderModel(labels={})
    assert set(verifier.score([("e", "c")])[0]) == {"contradiction", "entailment", "neutral"}
    verifier._model = FakeCrossEncoderModel(logits=[[1.0, 2.0]])
    with pytest.raises(VerifierUnavailableError):
        verifier.score([("e", "c")])
    verifier._model = FakeCrossEncoderModel(error=RuntimeError("predict failed"))
    with pytest.raises(VerifierUnavailableError):
        verifier.score([("e", "c")])

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("load failed")))
    verifier._model = None
    with pytest.raises(VerifierUnavailableError):
        verifier._load()


def test_default_verifier_is_cached_and_structured_generation_accepts_parsed(monkeypatch):
    monkeypatch.setattr(grounding_module, "_DEFAULT_VERIFIER", None)
    assert grounding_module.get_default_verifier() is grounding_module.get_default_verifier()
    parsed = StructuredDraft(answerable=True, claims=[DraftClaim(text="Parsed claim.", cited_chunk_ids=["c1"])])
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: SimpleNamespace(parsed=parsed, text="")))
    draft = generate_structured_draft("q", [hit("c1", "Parsed claim.")], client, "fake")
    assert draft.claims[0].text == "Parsed claim."
    assert generate_structured_draft("q", [], client, "fake").answerable is False


def test_structured_generation_fails_after_two_invalid_responses():
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: SimpleNamespace(parsed=None, text="bad")))
    with pytest.raises(ValueError):
        generate_structured_draft("q", [hit("c1", "Evidence.")], client, "fake")


def test_structured_generation_retries_unknown_citations():
    responses = iter([
        SimpleNamespace(parsed={"answerable": True, "claims": [{"text": "Claim.", "cited_chunk_ids": ["missing"]}]}, text=""),
        SimpleNamespace(parsed={"answerable": True, "claims": [{"text": "Claim.", "cited_chunk_ids": ["c1"]}]}, text=""),
    ])
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: next(responses)))
    assert generate_structured_draft("q", [hit("c1", "Claim.")], client, "fake").claims[0].cited_chunk_ids == ["c1"]
