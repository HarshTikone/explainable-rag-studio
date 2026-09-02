"""Headless quality-gate release orchestration and retained evidence artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .config import SETTINGS
from .contextual_chunking import build_contextual_chunks
from .document_parsers import parse_document
from .embeddings import Embedder
from .eval import run_eval
from .experiments import (
    EXPERIMENT_SCHEMA_VERSION, compare_experiments, create_experiment_config,
    dependency_versions, run_experiment, runtime_environment, save_comparison_artifact,
    source_tree_state,
)
from .grounding import (
    build_extractive_draft, citations_from_grounding, render_grounded_answer, verify_claims,
)
from .grounding_eval import (
    calibrate_grounding_policy, grounding_benchmark_fingerprint, grounding_promotion_gate,
    run_grounding_benchmark,
)
from .grounding_policy import GroundingPolicy, default_grounding_policy
from .reranker import get_default_reranker
from .retriever import retrieve
from .utils import ensure_dir, write_json
from .vectorstore import FaissStore


RELEASE_SCHEMA_VERSION = "1.1"


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hardware_manifest() -> Dict[str, Any]:
    memory_bytes = None
    try:
        if hasattr(os, "sysconf"):
            memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (OSError, TypeError, ValueError):
        pass
    return {
        **runtime_environment(),
        "cpu": platform.processor() or platform.machine(),
        "memory_bytes": memory_bytes,
        "model_state": "warmed",
        "measured_repeats": 3,
    }


def load_public_corpus(corpus_dir: str) -> List[Dict[str, Any]]:
    chunks = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        parsed = parse_document(str(path), source_name=path.name)
        chunks.extend(build_contextual_chunks(parsed.document, parsed.version, parsed.blocks))
    items = [chunk.to_dict() for chunk in chunks]
    if len(items) != 60:
        raise ValueError(f"Release corpus must produce exactly 60 deterministic chunks; got {len(items)}.")
    return items


def build_release_store(index_dir: str, items: List[Dict[str, Any]], embedder) -> FaissStore:
    vectors = embedder.embed_texts([item.get("retrieval_text", item["text"]) for item in items])
    store = FaissStore(index_dir)
    store.build(vectors, items)
    if store.index.ntotal != len(items) or len(store.meta["items"]) != len(items):
        raise ValueError("Release index vector/metadata alignment failed.")
    return store


def _retrieval_ask(store, embedder, strategy: str, repeats: int, reranker=None):
    def ask(question: str) -> Dict[str, Any]:
        runs = [retrieve(store, embedder, question, 5, strategy, reranker=reranker) for _ in range(repeats)]
        first = runs[0]
        hits = [hit.to_dict() for hit in first.hits]
        citations = [{"chunk_id": hits[0]["chunk_id"]}] if hits else []
        return {
            "answer": hits[0].get("text", "") if hits else "I don't know based on the retrieved evidence.",
            "citations": citations,
            "retrieved": hits,
            "retrieval_latency_ms": first.latency_ms,
            "retrieval_latency_samples_ms": [run.latency_ms for run in runs],
            "total_latency_ms": first.latency_ms,
            "total_latency_samples_ms": [run.latency_ms for run in runs],
            "stage_latency_ms": {
                "dense": first.dense_latency_ms, "lexical": first.lexical_latency_ms,
                "fusion": first.fusion_latency_ms, "reranking": first.reranking_latency_ms,
            },
            "stage_latency_samples_ms": {
                "dense": [run.dense_latency_ms for run in runs],
                "lexical": [run.lexical_latency_ms for run in runs],
                "fusion": [run.fusion_latency_ms for run in runs],
                "reranking": [run.reranking_latency_ms for run in runs],
            },
            "reranking_trace": first.reranking_trace,
        }
    return ask


def run_retrieval_release(
    benchmark: List[Dict[str, Any]], store, embedder, output_root: str,
    *, repeats: int = 3, reranker=None,
) -> Dict[str, Any]:
    reports = []
    for strategy in ("hybrid_rrf", "hybrid_rerank"):
        config = create_experiment_config(
            strategy=strategy, top_k=5, embedding_model=SETTINGS.embedding_model,
            chunk_tokens=SETTINGS.chunk_tokens, chunk_overlap=SETTINGS.chunk_overlap,
            corpus_items=store.meta["items"], benchmark_items=benchmark,
        )
        ask = _retrieval_ask(store, embedder, strategy, repeats, reranker if strategy == "hybrid_rerank" else None)
        reports.append(run_experiment(benchmark, ask, config, output_root))
    baseline, candidate = reports
    comparison = compare_experiments(baseline, candidate)
    artifact = save_comparison_artifact(baseline, candidate, comparison, output_root)
    selected = candidate if comparison["passed"] else baseline
    return {
        "baseline": baseline, "candidate": candidate, "comparison": comparison,
        "comparison_artifact": artifact, "selected_strategy": selected["config"]["strategy"],
        "selected": selected,
    }


def build_draft_bundles(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    bundles = []
    for row in report["results"]:
        hits = row.get("retrieval_hits", [])
        legacy = [(float(hit.get("final_score", 0.0)), hit) for hit in hits]
        draft = build_extractive_draft(legacy)
        bundles.append({
            "question": row["question"], "category": row.get("category", "general"),
            "retrieved": hits, "draft": draft.model_dump(),
        })
    return bundles


def _baseline_from_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    claims = bundle["draft"]["claims"]
    if not claims:
        answer = "I don't know based on the retrieved evidence."
    else:
        answer = " ".join(claim["text"] for claim in claims)
    citations = []
    seen = set()
    retrieved_ids = {item.get("chunk_id") for item in bundle["retrieved"]}
    for claim in claims:
        for chunk_id in claim["cited_chunk_ids"]:
            if chunk_id in retrieved_ids and chunk_id not in seen:
                citations.append({"chunk_id": chunk_id})
                seen.add(chunk_id)
    return {"answer": answer, "citations": citations, "retrieved": bundle["retrieved"]}


def run_grounding_qa(
    benchmark: List[Dict[str, Any]], bundles: List[Dict[str, Any]], verifier,
    policy: GroundingPolicy, output_dir: str, *, repeats: int = 3,
) -> Dict[str, Any]:
    from .grounding_models import StructuredDraft

    by_question = {bundle["question"]: bundle for bundle in bundles}
    baseline_outputs = {question: _baseline_from_bundle(bundle) for question, bundle in by_question.items()}
    strict_outputs: Dict[str, Dict[str, Any]] = {}
    for question, bundle in by_question.items():
        draft = StructuredDraft.model_validate(bundle["draft"])
        legacy = [(float(hit.get("final_score", 0.0)), hit) for hit in bundle["retrieved"]]
        started = time.perf_counter()
        runs = [verify_claims(draft, legacy, verifier=verifier, policy=policy).result for _ in range(repeats)]
        total_ms = (time.perf_counter() - started) * 1000
        first = runs[0]
        strict_outputs[question] = {
            "answer": render_grounded_answer(first), "citations": citations_from_grounding(first),
            "retrieved": bundle["retrieved"], "grounding": first.model_dump(),
            "verification_latency_samples_ms": [run.latency_ms["total_verification"] for run in runs],
            "total_latency_ms": first.latency_ms["total_verification"],
            "total_latency_samples_ms": [run.latency_ms["total_verification"] for run in runs],
            "measured_batch_wall_ms": total_ms,
        }
    ensure_dir(output_dir)
    baseline = run_eval(benchmark, lambda question: baseline_outputs[question], output_dir, write_report=False)
    strict = run_eval(benchmark, lambda question: strict_outputs[question], output_dir, write_report=False)
    draft_fingerprint = canonical_fingerprint(bundles)
    retrieval_fingerprint = canonical_fingerprint([
        {"question": bundle["question"], "retrieved": bundle["retrieved"]} for bundle in bundles
    ])
    for report, policy_name in ((baseline, "structured_unfiltered"), (strict, "strict_grounded")):
        report.update({
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "policy": policy_name,
            "draft_claims_fingerprint": draft_fingerprint,
            "retrieval_results_fingerprint": retrieval_fingerprint,
            "grounding_policy_id": policy.policy_id,
        })
    write_json(os.path.join(output_dir, "structured_unfiltered.json"), baseline)
    write_json(os.path.join(output_dir, "strict_grounded.json"), strict)
    write_json(os.path.join(output_dir, "draft_bundles.json"), bundles)
    return {"baseline": baseline, "strict": strict, "draft_fingerprint": draft_fingerprint, "retrieval_fingerprint": retrieval_fingerprint}


def _calibrate_locked_policy(calibration, items, verifier) -> Dict[str, Any]:
    if len(calibration) != 48 or any(case.get("split") != "calibration" for case in calibration):
        raise ValueError("Calibration requires exactly the frozen 48 calibration cases and cannot receive held-out cases.")
    result = calibrate_grounding_policy(calibration, items, verifier, default_grounding_policy())
    attempts = [{"model": verifier.model_name, "revision": verifier.model_revision, "result": result}]
    selected_verifier = verifier
    if not result["selected"] and verifier.model_name == SETTINGS.grounding_model:
        from .grounding import CrossEncoderNliVerifier

        fallback_model = SETTINGS.grounding_fallback_model
        fallback_revision = SETTINGS.grounding_fallback_revision
        selected_verifier = CrossEncoderNliVerifier(
            fallback_model, fallback_revision, SETTINGS.grounding_batch_size, SETTINGS.grounding_max_length,
            SETTINGS.grounding_backend, SETTINGS.grounding_onnx_file, SETTINGS.model_cpu_threads,
        )
        fallback_policy = replace(
            default_grounding_policy(), model_name=fallback_model, model_revision=fallback_revision
        )
        result = calibrate_grounding_policy(calibration, items, selected_verifier, fallback_policy)
        attempts.append({"model": fallback_model, "revision": fallback_revision, "result": result})
    if not result["selected"]:
        raise RuntimeError("No grounding policy met calibration safety constraints.")
    policy = GroundingPolicy.from_dict(result["selected"]["policy"])
    return {"policy": policy, "calibration": result, "verifier": selected_verifier, "attempts": attempts}


def run_release_validation(
    root: str, *, embedder=None, reranker=None, verifier=None, repeats: int = 3,
    runtime_checks: Dict[str, bool] | None = None, output_root: str | None = None,
) -> str:
    root_path = Path(root).resolve()
    release_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release_dir = Path(output_root).resolve() / release_id if output_root else root_path / SETTINGS.outputs_dir / "releases" / release_id
    ensure_dir(str(release_dir))
    items = load_public_corpus(str(root_path / "data" / "public_demo"))
    benchmark = json.loads((root_path / "data" / "public_demo_benchmark.json").read_text(encoding="utf-8"))
    cases = json.loads((root_path / "data" / "grounding_benchmark.json").read_text(encoding="utf-8"))
    calibration = [case for case in cases if case.get("split") == "calibration"]
    heldout = [case for case in cases if case.get("split") == "heldout"]
    if len(calibration) != 48 or len(heldout) != 48:
        raise ValueError("Grounding benchmark must contain frozen 48-case calibration and held-out splits.")
    active_embedder = embedder or Embedder(SETTINGS.embedding_model)
    store = build_release_store(str(release_dir / "index"), items, active_embedder)
    active_reranker = reranker or get_default_reranker(SETTINGS.reranker_model, SETTINGS.rerank_batch_size)
    active_verifier = verifier
    if active_verifier is None:
        from .grounding import get_default_verifier
        active_verifier = get_default_verifier()
    # Warm real models before measured passes.
    retrieve(store, active_embedder, benchmark[0]["question"], 5, "hybrid_rerank", reranker=active_reranker)
    active_verifier.score([("Warm verifier evidence.", "Warm verifier evidence.")])
    locked = _calibrate_locked_policy(calibration, items, active_verifier)
    policy = locked["policy"]
    active_verifier = locked["verifier"]
    retrieval = run_retrieval_release(
        benchmark, store, active_embedder, str(release_dir), repeats=repeats, reranker=active_reranker
    )
    bundles = build_draft_bundles(retrieval["selected"])
    qa = run_grounding_qa(
        benchmark, bundles, active_verifier, policy, str(release_dir / "grounding_qa"), repeats=repeats
    )
    heldout_report = run_grounding_benchmark(heldout, items, active_verifier, policy)
    grounding_gate = grounding_promotion_gate(heldout_report, qa["strict"], qa["baseline"])
    checks = runtime_checks or {}
    runtime_gates = {
        "python_3_11": sys.version_info[:2] == (3, 11),
        "docker_build": bool(checks.get("docker_build", False)),
        "streamlit_health": bool(checks.get("streamlit_health", False)),
        "api_smoke": bool(checks.get("api_smoke", False)),
        "dependency_check": bool(checks.get("dependency_check", False)),
    }
    tree = source_tree_state(str(root_path))
    retrieval_decision = {
        "closed": True, "promoted": bool(retrieval["comparison"]["passed"]),
        "retained_strategy": retrieval["selected_strategy"], "gates": retrieval["comparison"],
    }
    grounding_decision = {
        "closed": True, "promoted": bool(grounding_gate["passed"]),
        "retained_policy": policy.policy_id if grounding_gate["passed"] else "strict_safe_abstention",
        "gates": grounding_gate,
    }
    runtime_decision = {"closed": True, "promoted": all(runtime_gates.values()), "gates": runtime_gates}
    quality_gate = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": release_id,
        "retrieval": retrieval_decision, "grounding": grounding_decision, "runtime": runtime_decision,
        "overall": {
            "promoted": retrieval_decision["promoted"] and grounding_decision["promoted"] and runtime_decision["promoted"],
            "all_decisions_closed": True,
        },
    }
    manifest = {
        "schema_version": RELEASE_SCHEMA_VERSION, "release_id": release_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_schema": EXPERIMENT_SCHEMA_VERSION,
        "source_tree": tree, "dependencies": dependency_versions(),
        "corpus_fingerprint": retrieval["baseline"]["config"]["corpus_fingerprint"],
        "benchmark_fingerprint": retrieval["baseline"]["config"]["benchmark_fingerprint"],
        "grounding_calibration_fingerprint": locked["calibration"]["calibration_fingerprint"],
        "grounding_heldout_fingerprint": grounding_benchmark_fingerprint(heldout, items),
        "draft_claims_fingerprint": qa["draft_fingerprint"],
        "retrieval_results_fingerprint": qa["retrieval_fingerprint"],
        "model_warm_state": "warmed", "measured_repeats": repeats,
    }
    write_json(str(release_dir / "manifest.json"), manifest)
    write_json(str(release_dir / "hardware.json"), hardware_manifest())
    write_json(str(release_dir / "grounding_policy.json"), {
        "schema_version": "1.0", "locked": True, "policy": policy.to_dict(),
        "calibration_fingerprint": locked["calibration"]["calibration_fingerprint"],
    })
    write_json(str(release_dir / "calibration_report.json"), locked["calibration"])
    write_json(str(release_dir / "calibration_attempts.json"), locked["attempts"])
    write_json(str(release_dir / "heldout_claim_report.json"), heldout_report)
    write_json(str(release_dir / "quality_gate.json"), quality_gate)
    return str(release_dir)


def validate_release_artifact(release_dir: str) -> Dict[str, Any]:
    path = Path(release_dir)
    required = (
        "manifest.json", "hardware.json", "grounding_policy.json", "calibration_report.json",
        "heldout_claim_report.json", "quality_gate.json", "grounding_qa/draft_bundles.json",
        "grounding_qa/structured_unfiltered.json", "grounding_qa/strict_grounded.json",
    )
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise ValueError(f"Release artifact is incomplete: {missing}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    policy_wrapper = json.loads((path / "grounding_policy.json").read_text(encoding="utf-8"))
    policy = GroundingPolicy.from_dict(policy_wrapper["policy"])
    bundles = json.loads((path / "grounding_qa" / "draft_bundles.json").read_text(encoding="utf-8"))
    if canonical_fingerprint(bundles) != manifest["draft_claims_fingerprint"]:
        raise ValueError("Draft-claim fingerprint mismatch.")
    if policy.policy_id != policy_wrapper["policy"]["policy_id"]:
        raise ValueError("Grounding policy fingerprint mismatch.")
    return {
        "valid": True, "release_id": manifest["release_id"],
        "quality_gate": json.loads((path / "quality_gate.json").read_text(encoding="utf-8")),
    }


def update_release_runtime_checks(release_dir: str, checks: Dict[str, bool]) -> Dict[str, Any]:
    """Attach independently verified runtime checks without rerunning model experiments."""
    validation = validate_release_artifact(release_dir)
    path = Path(release_dir) / "quality_gate.json"
    quality_gate = validation["quality_gate"]
    runtime_gates = quality_gate["runtime"]["gates"]
    runtime_gates["python_3_11"] = sys.version_info[:2] == (3, 11)
    for key in ("docker_build", "streamlit_health", "api_smoke", "dependency_check"):
        if key in checks:
            runtime_gates[key] = bool(checks[key])
    quality_gate["runtime"] = {
        "closed": True, "promoted": all(runtime_gates.values()), "gates": runtime_gates,
    }
    quality_gate["overall"]["promoted"] = bool(
        quality_gate["retrieval"]["promoted"] and quality_gate["grounding"]["promoted"] and quality_gate["runtime"]["promoted"]
    )
    write_json(str(path), quality_gate)
    return quality_gate


def write_portfolio_summary(release_dir: str, target: str) -> str:
    validation = validate_release_artifact(release_dir)
    path = Path(release_dir)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    heldout = json.loads((path / "heldout_claim_report.json").read_text(encoding="utf-8"))
    strict = json.loads((path / "grounding_qa" / "strict_grounded.json").read_text(encoding="utf-8"))
    baseline_qa = json.loads((path / "grounding_qa" / "structured_unfiltered.json").read_text(encoding="utf-8"))
    experiment_reports = []
    for report_path in sorted((path / "experiments").glob("*/report.json")):
        experiment_reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    retrieval = {
        report["config"]["strategy"]: {
            "recall_at_5": report["retrieval"]["recall_at_5"],
            "mrr": report["retrieval"]["mrr"],
            "ndcg_at_5": report["retrieval"]["ndcg_at_5"],
            "retrieval_p95_ms": report["latency_ms"]["retrieval_p95"],
        }
        for report in experiment_reports
    }
    summary = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": manifest["release_id"],
        "source_tree_fingerprint": manifest["source_tree"]["source_tree_fingerprint"],
        "quality_gate": validation["quality_gate"],
        "grounding_heldout": {
            "macro_f1": heldout["macro_f1"],
            "supported_precision": heldout["per_label"]["supported"]["precision"],
            "contradiction_recall": heldout["per_label"]["contradiction"]["recall"],
        },
        "retrieval": retrieval,
        "strict_qa": {
            "accuracy": strict["accuracy"], "baseline_accuracy": baseline_qa["accuracy"],
            "abstention_accuracy": strict["abstention_accuracy"],
            "baseline_abstention_accuracy": baseline_qa["abstention_accuracy"],
            "answer_coverage": strict["grounding"]["answer_coverage"],
            "citation_validity": strict["citation_validity"],
            "claim_citation_coverage": strict["grounding"]["claim_citation_coverage"],
            "verification_p95_ms": strict["latency_ms"]["verification_p95"],
        },
    }
    ensure_dir(str(Path(target).parent))
    write_json(target, summary)
    return target
