import subprocess
from dataclasses import replace
from pathlib import Path

from backend.experiments import (
    benchmark_fingerprint,
    compare_experiments,
    corpus_fingerprint,
    create_experiment_config,
    run_experiment,
    save_comparison_artifact,
    source_tree_state,
)


CORPUS = [{"chunk_id": "c1", "source": "demo.md", "page": 1, "text": "  Stable   text "}]
BENCHMARK = [{"question": "Q?", "expected": "answer", "relevant_chunk_ids": ["c1"]}]


def test_fingerprints_are_stable_for_normalized_corpus():
    equivalent = [{"chunk_id": "c1", "source": "demo.md", "page": 1, "text": "Stable text"}]
    assert corpus_fingerprint(CORPUS) == corpus_fingerprint(equivalent)
    assert benchmark_fingerprint(BENCHMARK) == benchmark_fingerprint(list(BENCHMARK))
    assert "source_tree_fingerprint" in source_tree_state()


def test_source_tree_fingerprint_ignores_generated_quality_reference(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    source = tmp_path / "backend" / "module.py"
    reference = tmp_path / "docs" / "benchmarks" / "quality-gate-reference.json"
    source.parent.mkdir(parents=True)
    reference.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    reference.write_text('{"overall": false}\n', encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    before = source_tree_state(str(tmp_path))["source_tree_fingerprint"]
    reference.write_text('{"overall": true}\n', encoding="utf-8")
    after_reference_change = source_tree_state(str(tmp_path))["source_tree_fingerprint"]
    source.write_text("VALUE = 2\n", encoding="utf-8")
    after_source_change = source_tree_state(str(tmp_path))["source_tree_fingerprint"]

    assert after_reference_change == before
    assert after_source_change != before


def test_experiment_id_and_artifacts(tmp_path):
    config = create_experiment_config(strategy="dense", top_k=5, embedding_model="fake", chunk_tokens=420, chunk_overlap=80, corpus_items=CORPUS, benchmark_items=BENCHMARK)

    def ask(_question):
        return {"answer": "answer", "citations": [{"chunk_id": "c1"}], "retrieved": [{"chunk_id": "c1"}], "retrieval_latency_ms": 2, "total_latency_ms": 3}

    report = run_experiment(BENCHMARK, ask, config, str(tmp_path))
    experiment_dir = tmp_path / "experiments" / config.experiment_id
    assert report["schema_version"] == "3.3"
    assert report["config"]["grounding_model"] == "cross-encoder/nli-deberta-v3-xsmall"
    assert report["config"]["dependency_versions"]
    assert (experiment_dir / "config.json").exists()
    assert (experiment_dir / "report.json").exists()
    assert (experiment_dir / "results.json").exists()


def test_comparison_gate_requires_same_inputs_and_measured_improvement():
    shared = {"corpus_fingerprint": "c", "benchmark_fingerprint": "b", "embedding_model": "e", "chunk_tokens": 420, "chunk_overlap": 80, "top_k": 5}
    base = {"config": shared, "retrieval": {"recall": .8, "mrr": .7, "ndcg": .70}, "citation_validity": 1.0, "latency_ms": {"retrieval_p95": 100}}
    candidate = {"config": dict(base["config"]), "retrieval": {"recall": .80, "mrr": .74, "ndcg": .72}, "citation_validity": 1.0, "latency_ms": {"retrieval_p95": 120}}
    assert compare_experiments(base, candidate)["passed"]
    candidate["config"] = {"corpus_fingerprint": "different", "benchmark_fingerprint": "b"}
    assert not compare_experiments(base, candidate)["passed"]
    candidate["config"] = {**shared, "embedding_model": "different"}
    assert not compare_experiments(base, candidate)["passed"]


def test_reranker_manifest_records_model_configuration():
    config = create_experiment_config(strategy="hybrid_rerank", top_k=5, embedding_model="fake", chunk_tokens=420, chunk_overlap=80, corpus_items=CORPUS, benchmark_items=BENCHMARK)
    assert config.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.rerank_candidates == 30


def test_mocked_sixty_question_schema_33_experiment_and_comparison(tmp_path):
    benchmark = [
        {"question": f"Question {index}", "expected": "answer", "relevant_chunk_ids": ["c1"], "category": "exact_term"}
        for index in range(60)
    ]

    def ask(_question):
        return {
            "answer": "answer",
            "citations": [{"chunk_id": "c1"}],
            "retrieved": [{"chunk_id": "c1", "fusion_rank": 1, "reranker_rank": 1}],
            "retrieval_latency_ms": 4,
            "total_latency_ms": 5,
            "stage_latency_ms": {"dense": 1, "lexical": 1, "fusion": 1, "reranking": 1},
        }

    baseline_config = create_experiment_config(strategy="hybrid_rrf", top_k=5, embedding_model="fake", chunk_tokens=420, chunk_overlap=80, corpus_items=CORPUS, benchmark_items=benchmark)
    candidate_config = create_experiment_config(strategy="hybrid_rerank", top_k=5, embedding_model="fake", chunk_tokens=420, chunk_overlap=80, corpus_items=CORPUS, benchmark_items=benchmark)
    baseline = run_experiment(benchmark, ask, baseline_config, str(tmp_path))
    candidate = run_experiment(benchmark, ask, candidate_config, str(tmp_path))
    comparison = compare_experiments(baseline, candidate)
    artifact = save_comparison_artifact(baseline, candidate, comparison, str(tmp_path))
    assert len(candidate["results"]) == 60
    assert candidate["schema_version"] == "3.3"
    assert Path(artifact).exists()
