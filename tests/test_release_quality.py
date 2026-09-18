import json
import hashlib
from pathlib import Path

import numpy as np

from backend.release_quality import (
    run_release_validation, update_release_runtime_checks, validate_release_artifact, write_portfolio_summary,
)


class FakeEmbedder:
    def embed_texts(self, texts):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            row = np.frombuffer(digest, dtype=np.uint8).astype("float32") - 127.5
            row /= np.linalg.norm(row)
            rows.append(row)
        return np.asarray(rows, dtype="float32")

    def embed_query(self, text):
        return self.embed_texts([text])


class FakeReranker:
    model_name = "fake-reranker"

    def score(self, query, documents):
        return [float(len(document) % 17) for document in documents]


class BenchmarkVerifier:
    model_name = "fake-verifier"
    model_revision = "test"

    def __init__(self, labels):
        self.labels = labels

    def score(self, pairs):
        rows = []
        for _premise, claim in pairs:
            label = self.labels.get(claim, "supported")
            if label == "contradiction":
                rows.append({"contradiction": .98, "entailment": .01, "neutral": .01})
            elif label == "neutral":
                rows.append({"contradiction": .01, "entailment": .01, "neutral": .98})
            else:
                rows.append({"contradiction": .01, "entailment": .98, "neutral": .01})
        return rows


def test_mocked_release_writes_and_validates_schema_33_artifact(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "data" / "grounding_benchmark.json").read_text(encoding="utf-8"))
    verifier = BenchmarkVerifier({case["claim"]: case["label"] for case in cases})
    release_dir = run_release_validation(
        str(root), embedder=FakeEmbedder(), reranker=FakeReranker(), verifier=verifier,
        repeats=1, output_root=str(tmp_path / "releases"),
        runtime_checks={"docker_build": True, "streamlit_health": True, "api_smoke": True, "dependency_check": True},
    )
    result = validate_release_artifact(release_dir)
    assert result["valid"]
    updated = update_release_runtime_checks(release_dir, {
        "docker_build": True, "streamlit_health": True, "api_smoke": True, "dependency_check": True,
    })
    assert updated["runtime"]["gates"]["api_smoke"] is True
    strict = json.loads((Path(release_dir) / "grounding_qa" / "strict_grounded.json").read_text(encoding="utf-8"))
    baseline = json.loads((Path(release_dir) / "grounding_qa" / "structured_unfiltered.json").read_text(encoding="utf-8"))
    assert strict["schema_version"] == "3.3"
    assert strict["draft_claims_fingerprint"] == baseline["draft_claims_fingerprint"]
    target = tmp_path / "portfolio.json"
    assert Path(write_portfolio_summary(release_dir, str(target))).exists()
