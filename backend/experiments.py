"""Reproducible experiment manifests and baseline/candidate comparison."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List
from pathlib import Path

from .eval import run_eval
from .utils import ensure_dir, write_json

from .config import SETTINGS
from .grounding_policy import default_grounding_policy

EXPERIMENT_SCHEMA_VERSION = "3.3"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def corpus_fingerprint(items: List[Dict[str, Any]]) -> str:
    canonical = [
        {
            "chunk_id": item.get("chunk_id", ""),
            "source": item.get("source", ""),
            "page": item.get("page"),
            "text": " ".join(str(item.get("text", "")).split()),
            "retrieval_text": " ".join(str(item.get("retrieval_text", "")).split()),
            "document_id": item.get("document_id", ""),
            "document_version_id": item.get("document_version_id", ""),
            "parent_id": item.get("parent_id", ""),
            "heading_path": item.get("heading_path", []),
            "content_fingerprint": item.get("content_fingerprint", ""),
        }
        for item in sorted(items, key=lambda value: str(value.get("chunk_id", "")))
    ]
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def benchmark_fingerprint(items: List[Dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(items).encode("utf-8")).hexdigest()


def git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def source_tree_state(root: str | None = None) -> Dict[str, Any]:
    """Fingerprint the effective source tree, including uncommitted files."""
    base = Path(root or Path(__file__).resolve().parents[1])
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=base, capture_output=True, text=True,
            check=True, timeout=5,
        ).stdout
        listed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=base,
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return {"dirty": None, "source_tree_fingerprint": "unknown"}
    digest = hashlib.sha256()
    for relative in sorted(value.strip() for value in listed if value.strip()):
        path = base / relative
        if not path.is_file() or relative.startswith(("outputs/", "index/", ".venv/")):
            continue
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"dirty": bool(status.strip()), "source_tree_fingerprint": digest.hexdigest()}


def runtime_environment() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "thread_env": {
            key: os.getenv(key, "") for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS")
        },
        "executable": sys.executable,
    }


def dependency_versions() -> Dict[str, str]:
    result = {}
    for package in ("faiss-cpu", "rank-bm25", "sentence-transformers", "sentencepiece", "numpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unknown"
    return result


@dataclass(frozen=True)
class ExperimentConfig:
    strategy: str
    top_k: int
    candidate_count: int
    embedding_model: str
    chunk_tokens: int
    chunk_overlap: int
    corpus_fingerprint: str
    benchmark_fingerprint: str
    git_revision: str
    created_at: str
    reranker_model: str | None
    rerank_candidates: int
    rerank_batch_size: int
    dependency_versions: Dict[str, str]
    source_tree_fingerprint: str = "unknown"
    git_dirty: bool | None = None
    runtime_environment: Dict[str, Any] | None = None
    grounding_policy_id: str = ""
    grounding_premise_version: str = ""
    grounding_guard_version: str = ""
    grounding_policy: str = "strict"
    grounding_model: str = SETTINGS.grounding_model
    grounding_model_revision: str = SETTINGS.grounding_model_revision
    grounding_entailment_threshold: float = SETTINGS.grounding_entailment_threshold
    grounding_contradiction_threshold: float = SETTINGS.grounding_contradiction_threshold
    grounding_prompt_version: str = SETTINGS.grounding_prompt_version
    schema_version: str = EXPERIMENT_SCHEMA_VERSION

    @property
    def experiment_id(self) -> str:
        digest = hashlib.sha256(_canonical_json(asdict(self)).encode("utf-8")).hexdigest()[:12]
        stamp = self.created_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
        return f"{stamp}-{self.strategy}-{digest}"


def create_experiment_config(
    *, strategy: str, top_k: int, embedding_model: str, chunk_tokens: int,
    chunk_overlap: int, corpus_items: List[Dict[str, Any]], benchmark_items: List[Dict[str, Any]],
    reranker_model: str | None = None, rerank_candidates: int | None = None,
    rerank_batch_size: int | None = None,
) -> ExperimentConfig:
    uses_reranker = strategy == "hybrid_rerank"
    tree = source_tree_state()
    policy = default_grounding_policy()
    return ExperimentConfig(
        strategy=strategy,
        top_k=top_k,
        candidate_count=max(50, top_k * 5),
        embedding_model=embedding_model,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
        corpus_fingerprint=corpus_fingerprint(corpus_items),
        benchmark_fingerprint=benchmark_fingerprint(benchmark_items),
        git_revision=git_revision(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        reranker_model=(reranker_model or SETTINGS.reranker_model) if uses_reranker else None,
        rerank_candidates=(rerank_candidates or SETTINGS.rerank_candidates) if uses_reranker else 0,
        rerank_batch_size=(rerank_batch_size or SETTINGS.rerank_batch_size) if uses_reranker else 0,
        dependency_versions=dependency_versions(),
        source_tree_fingerprint=tree["source_tree_fingerprint"],
        git_dirty=tree["dirty"],
        runtime_environment=runtime_environment(),
        grounding_policy_id=policy.policy_id,
        grounding_premise_version=policy.premise_version,
        grounding_guard_version=policy.guard_version,
        grounding_policy="strict",
        grounding_model=SETTINGS.grounding_model,
        grounding_model_revision=SETTINGS.grounding_model_revision,
        grounding_entailment_threshold=SETTINGS.grounding_entailment_threshold,
        grounding_contradiction_threshold=SETTINGS.grounding_contradiction_threshold,
        grounding_prompt_version=SETTINGS.grounding_prompt_version,
    )


def run_experiment(eval_items, ask_fn, config: ExperimentConfig, output_root: str) -> Dict[str, Any]:
    experiment_dir = os.path.join(output_root, "experiments", config.experiment_id)
    ensure_dir(experiment_dir)
    report = run_eval(eval_items, ask_fn, experiment_dir, write_report=False)
    results = report.pop("results")
    report.update({"schema_version": EXPERIMENT_SCHEMA_VERSION, "experiment_id": config.experiment_id, "config": asdict(config)})
    write_json(os.path.join(experiment_dir, "config.json"), asdict(config))
    write_json(os.path.join(experiment_dir, "report.json"), report)
    write_json(os.path.join(experiment_dir, "results.json"), results)
    return {**report, "results": results, "experiment_dir": experiment_dir}


def compare_experiments(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    comparable_keys = (
        "corpus_fingerprint", "benchmark_fingerprint", "embedding_model", "chunk_tokens",
        "chunk_overlap", "top_k", "source_tree_fingerprint", "grounding_policy_id",
    )
    same_inputs = all(baseline["config"].get(key) == candidate["config"].get(key) for key in comparable_keys)
    base_recall = baseline["retrieval"].get("recall_at_5", baseline["retrieval"]["recall"])
    candidate_recall = candidate["retrieval"].get("recall_at_5", candidate["retrieval"]["recall"])
    base_mrr, candidate_mrr = baseline["retrieval"]["mrr"], candidate["retrieval"]["mrr"]
    base_ndcg = baseline["retrieval"].get("ndcg_at_5", baseline["retrieval"].get("ndcg"))
    candidate_ndcg = candidate["retrieval"].get("ndcg_at_5", candidate["retrieval"].get("ndcg"))
    labeled = None not in (base_recall, candidate_recall, base_mrr, candidate_mrr, base_ndcg, candidate_ndcg)
    recall_delta = (candidate_recall - base_recall) if labeled else None
    mrr_delta = (candidate_mrr - base_mrr) if labeled else None
    ndcg_delta = (candidate_ndcg - base_ndcg) if labeled else None
    base_p95 = baseline["latency_ms"]["retrieval_p95"] or 0.0
    candidate_p95 = candidate["latency_ms"]["retrieval_p95"] or 0.0
    latency_ok = candidate_p95 < 1500.0
    quality_ok = bool(labeled and max(mrr_delta, ndcg_delta) >= 0.03 and recall_delta >= -0.01)
    citation_ok = candidate["citation_validity"] == 1.0
    return {
        "passed": same_inputs and quality_ok and citation_ok and latency_ok,
        "same_inputs": same_inputs,
        "quality_labeled": labeled,
        "recall_delta": recall_delta,
        "mrr_delta": mrr_delta,
        "ndcg_delta": ndcg_delta,
        "retrieval_p95_delta_ms": candidate_p95 - base_p95,
        "quality_ok": quality_ok,
        "citation_ok": citation_ok,
        "latency_ok": latency_ok,
    }


def save_comparison_artifact(
    baseline: Dict[str, Any], candidate: Dict[str, Any], comparison: Dict[str, Any], output_root: str
) -> str:
    """Persist a portfolio-ready summary with category and question-level movement."""
    baseline_by_question = {result["question"]: result for result in baseline["results"]}
    movements = []
    for result in candidate["results"]:
        previous = baseline_by_question[result["question"]]
        movements.append({
            "question": result["question"],
            "category": result["category"],
            "baseline_chunks": previous["retrieved_chunk_ids"],
            "candidate_chunks": result["retrieved_chunk_ids"],
            "mrr_delta": None if result["reciprocal_rank"] is None else result["reciprocal_rank"] - previous["reciprocal_rank"],
            "ndcg_delta": None if result["ndcg"] is None else result["ndcg"] - previous["ndcg"],
        })
    ranked_movements = sorted(
        movements,
        key=lambda row: -abs((row["ndcg_delta"] or 0.0) + (row["mrr_delta"] or 0.0)),
    )
    improvements = sorted(
        movements,
        key=lambda row: -((row["ndcg_delta"] or 0.0) + (row["mrr_delta"] or 0.0)),
    )
    regressions = sorted(
        movements,
        key=lambda row: ((row["ndcg_delta"] or 0.0) + (row["mrr_delta"] or 0.0)),
    )
    categories = sorted({result["category"] for result in baseline["results"]})
    category_summary = []
    for category in categories:
        for label, report in (("baseline", baseline), ("candidate", candidate)):
            rows = [result for result in report["results"] if result["category"] == category]
            category_summary.append({
                "category": category,
                "pipeline": label,
                "n": len(rows),
                "recall": _mean([row["recall"] for row in rows]),
                "mrr": _mean([row["reciprocal_rank"] for row in rows]),
                "ndcg": _mean([row["ndcg"] for row in rows]),
            })
    artifact = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "baseline_experiment_id": baseline["experiment_id"],
        "candidate_experiment_id": candidate["experiment_id"],
        "comparison": comparison,
        "baseline_metrics": {"retrieval": baseline["retrieval"], "grounding": baseline.get("grounding", {}), "latency_ms": baseline["latency_ms"], "stage_latency_ms": baseline["stage_latency_ms"]},
        "candidate_metrics": {"retrieval": candidate["retrieval"], "grounding": candidate.get("grounding", {}), "latency_ms": candidate["latency_ms"], "stage_latency_ms": candidate["stage_latency_ms"]},
        "category_summary": category_summary,
        "largest_movements": ranked_movements[:10],
        "largest_improvements": improvements[:10],
        "largest_regressions": regressions[:10],
    }
    comparison_dir = os.path.join(output_root, "comparisons")
    ensure_dir(comparison_dir)
    path = os.path.join(comparison_dir, f"{baseline['experiment_id']}__vs__{candidate['experiment_id']}.json")
    write_json(path, artifact)
    return path


def _mean(values):
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None
