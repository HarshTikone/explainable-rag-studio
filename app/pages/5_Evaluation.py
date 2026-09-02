import json
import time

import pandas as pd
import streamlit as st
from google import genai

from app.ui import configure_page, footer, page_header, section, status_pills
from backend.config import SETTINGS
from backend.embeddings import Embedder
from backend.eval import validate_eval_set
from backend.experiments import compare_experiments, create_experiment_config, run_experiment, save_comparison_artifact
from backend.grounding_eval import compare_grounding_policies
from backend.grounding_policy import default_grounding_policy
from backend.qa import answer_with_optional_llm
from backend.retriever import retrieve
from backend.reranker import RerankerUnavailableError
from backend.vectorstore import FaissStore
from app.security_ui import security_context, tenant_store

configure_page("Evaluation", "✓")
security = security_context("experiments:run")
page_header("Quality lab", "Prove which retrieval pipeline wins.", "Run a reproducible baseline and candidate against the same labeled benchmark, then inspect quality, latency, and question-level ranking changes.")

store = tenant_store(security)
if store.index is None:
    st.warning("No index found. Build one first in Ingest & Index.")
    st.stop()

policy = default_grounding_policy()
status_pills([("Index online", True), ("Experiment schema v3.3", True), ("Strict claim grounding", True)])
st.caption(
    f"Grounding policy `{policy.policy_id}` · premise `{policy.premise_version}` · "
    f"guards `{policy.guard_version}`. Calibration uses only the 48 calibration claims; held-out labels remain sealed."
)
with st.expander("Evaluation set format"):
    st.code('[\n  {\n    "question": "...",\n    "reference_answer": "...",\n    "relevant_chunk_ids": ["c000001"],\n    "answerable": true,\n    "category": "exact_term"\n  }\n]', language="json")
    st.caption("Gold chunk IDs unlock Recall@k, Precision@k, hit rate, and MRR. Legacy question/expected files remain supported.")

eval_file = st.file_uploader("Upload a labeled benchmark", type=["json"])
section("Experiment configuration", "Baseline and candidate always run against the same index fingerprint and benchmark payload.")
c1, c2, c3, c4 = st.columns(4)
with c1:
    baseline_strategy = st.selectbox("Baseline", ["hybrid_rrf", "dense_mmr", "dense", "hybrid_rerank"], index=0)
with c2:
    candidate_strategy = st.selectbox("Candidate", ["hybrid_rerank", "hybrid_rrf", "dense_mmr", "dense"], index=0)
with c3:
    top_k = st.slider("Top-K", 3, 12, SETTINGS.top_k, 1)
with c4:
    embed_model = st.text_input("Embedding model", SETTINGS.embedding_model)

use_gemini = False
gemini_client = None
if SETTINGS.gemini_api_key.strip():
    try:
        gemini_client = genai.Client(api_key=SETTINGS.gemini_api_key)
        use_gemini = True
    except Exception:
        pass
st.caption(f"Generator: {SETTINGS.gemini_model if use_gemini else 'deterministic extractive fallback'}")


def make_ask_fn(embedder, strategy):
    def ask(question: str):
        started = time.perf_counter()
        retrieval = retrieve(store, embedder, question, top_k, strategy)
        output = answer_with_optional_llm(
            question=question,
            retrieved_items=retrieval,
            use_gemini=use_gemini,
            gemini_client=gemini_client,
            gemini_model=SETTINGS.gemini_model,
            persist_review=False,
        )
        output["retrieved"] = [hit.to_dict() for hit in retrieval.hits]
        output["retrieval_latency_ms"] = retrieval.latency_ms
        output["total_latency_ms"] = (time.perf_counter() - started) * 1000
        output["stage_latency_ms"] = {
            "dense": retrieval.dense_latency_ms,
            "lexical": retrieval.lexical_latency_ms,
            "fusion": retrieval.fusion_latency_ms,
            "reranking": retrieval.reranking_latency_ms,
        }
        output["reranking_trace"] = retrieval.reranking_trace
        return output
    return ask


if st.button("Run baseline vs candidate", type="primary", disabled=not eval_file, use_container_width=True):
    try:
        eval_items = json.loads(eval_file.getvalue().decode("utf-8"))
        validate_eval_set(eval_items)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Invalid evaluation set: {exc}")
        st.stop()
    if baseline_strategy == candidate_strategy:
        st.error("Choose different baseline and candidate strategies.")
        st.stop()

    embedder = Embedder(embed_model)
    reports = []
    with st.status("Running reproducible experiments…", expanded=True) as run_status:
        try:
            for label, strategy in (("Baseline", baseline_strategy), ("Candidate", candidate_strategy)):
                st.write(f"{label}: `{strategy}`")
                config = create_experiment_config(
                    strategy=strategy,
                    top_k=top_k,
                    embedding_model=embed_model,
                    chunk_tokens=SETTINGS.chunk_tokens,
                    chunk_overlap=SETTINGS.chunk_overlap,
                    corpus_items=store.meta["items"],
                    benchmark_items=eval_items,
                )
                reports.append(run_experiment(eval_items, make_ask_fn(embedder, strategy), config, SETTINGS.outputs_dir))
            run_status.update(label="Both experiments completed", state="complete", expanded=False)
        except RerankerUnavailableError as exc:
            run_status.update(label="Reranker unavailable", state="error", expanded=True)
            st.error(f"The local reranker could not be loaded: {exc}")
            st.stop()

    baseline, candidate = reports
    comparison = compare_experiments(baseline, candidate)
    section("Promotion gate", "The candidate must improve retrieval quality, preserve citations, and stay inside the latency budget.")
    if comparison["passed"]:
        st.success("PROMOTED — candidate passed every retrieval promotion gate.")
    else:
        st.warning("REJECTED — retain the baseline strategy and review the failed gates below.")

    gate_columns = st.columns(4)
    for column, (label, key) in zip(gate_columns, (("Same inputs", "same_inputs"), ("Quality", "quality_ok"), ("Citations", "citation_ok"), ("Latency", "latency_ok"))):
        column.metric(label, "Pass" if comparison[key] else "Fail")

    section("Aggregate comparison")
    recall_delta = comparison["recall_delta"]
    mrr_delta = comparison["mrr_delta"]
    ndcg_delta = comparison["ndcg_delta"]
    metrics = pd.DataFrame([
        {"metric": "Recall@5", "baseline": baseline["retrieval"]["recall_at_5"], "candidate": candidate["retrieval"]["recall_at_5"], "delta": recall_delta},
        {"metric": "MRR", "baseline": baseline["retrieval"]["mrr"], "candidate": candidate["retrieval"]["mrr"], "delta": mrr_delta},
        {"metric": "nDCG@5", "baseline": baseline["retrieval"]["ndcg_at_5"], "candidate": candidate["retrieval"]["ndcg_at_5"], "delta": ndcg_delta},
        {"metric": "nDCG@k", "baseline": baseline["retrieval"]["ndcg"], "candidate": candidate["retrieval"]["ndcg"], "delta": candidate["retrieval"]["ndcg"] - baseline["retrieval"]["ndcg"]},
        {"metric": "Citation validity", "baseline": baseline["citation_validity"], "candidate": candidate["citation_validity"], "delta": candidate["citation_validity"] - baseline["citation_validity"]},
        {"metric": "Retrieval p95 ms", "baseline": baseline["latency_ms"]["retrieval_p95"], "candidate": candidate["latency_ms"]["retrieval_p95"], "delta": comparison["retrieval_p95_delta_ms"]},
        {"metric": "Displayed claim support", "baseline": baseline["grounding"]["displayed_claim_support"], "candidate": candidate["grounding"]["displayed_claim_support"], "delta": (candidate["grounding"]["displayed_claim_support"] or 0) - (baseline["grounding"]["displayed_claim_support"] or 0)},
        {"metric": "Answer coverage", "baseline": baseline["grounding"]["answer_coverage"], "candidate": candidate["grounding"]["answer_coverage"], "delta": (candidate["grounding"]["answer_coverage"] or 0) - (baseline["grounding"]["answer_coverage"] or 0)},
        {"metric": "Verification p95 ms", "baseline": baseline["latency_ms"]["verification_p95"], "candidate": candidate["latency_ms"]["verification_p95"], "delta": (candidate["latency_ms"]["verification_p95"] or 0) - (baseline["latency_ms"]["verification_p95"] or 0)},
    ])
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    section("Category slices", "Mean quality by benchmark category.")
    category_frames = []
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        frame = pd.DataFrame(report["results"])
        grouped = frame.groupby("category", dropna=False)[["score", "recall", "recall_at_5", "reciprocal_rank", "ndcg", "ndcg_at_5", "abstention_correct"]].mean(numeric_only=True).reset_index()
        grouped.insert(1, "pipeline", label)
        category_frames.append(grouped)
    st.dataframe(pd.concat(category_frames, ignore_index=True), use_container_width=True, hide_index=True)

    section("Grounding impact", "Strict rendering exposes only supported claims and records every removed or disputed claim.")
    st.caption(
        "Answer coverage is measured over answerable questions only. Citation validity is aggregated over displayed "
        "citations, so a correct citation-free abstention is not treated as invalid."
    )
    grounding_rows = []
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        grounding_rows.append({
            "pipeline": label,
            "displayed_claim_support": report["grounding"]["displayed_claim_support"],
            "claim_citation_coverage": report["grounding"]["claim_citation_coverage"],
            "answer_coverage": report["grounding"]["answer_coverage"],
            "accepted_claims": report["grounding"]["accepted_claims"],
            "removed_claims": report["grounding"]["rejected_claims"],
            "conflicts": report["grounding"]["conflicts"],
            "verification_p95_ms": report["latency_ms"]["verification_p95"],
        })
    st.dataframe(pd.DataFrame(grounding_rows), use_container_width=True, hide_index=True)
    failure_rows = []
    for result in candidate["results"]:
        for claim in result.get("grounding", {}).get("rejected_claims", []):
            failure_rows.append({
                "question": result["question"], "category": result["category"],
                "claim": claim["text"], "verdict": claim["verdict"],
                "entailment": claim["entailment_score"], "contradiction": claim["contradiction_score"],
                "reasons": ", ".join(claim["reason_codes"]),
            })
    if failure_rows:
        st.dataframe(pd.DataFrame(failure_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No claims were removed in the candidate run.")
    policy_comparison = compare_grounding_policies(candidate)
    st.caption("Policy comparison reuses the candidate's exact retrieved evidence and cached draft claims; only rendering policy changes.")
    st.dataframe(pd.DataFrame([
        {"policy": "structured_unfiltered", **policy_comparison["structured_unfiltered"]},
        {"policy": "strict_grounded", **policy_comparison["strict_grounded"]},
    ]), use_container_width=True, hide_index=True)

    section("Question-level differences", "Compare retrieved chunk IDs and ranking outcomes for each question.")
    rows = []
    for base_result, candidate_result in zip(baseline["results"], candidate["results"]):
        rows.append({
            "question": base_result["question"],
            "category": base_result["category"],
            "baseline_chunks": ", ".join(base_result["retrieved_chunk_ids"]),
            "candidate_chunks": ", ".join(candidate_result["retrieved_chunk_ids"]),
            "recall_delta": None if base_result["recall"] is None else candidate_result["recall"] - base_result["recall"],
            "mrr_delta": None if base_result["reciprocal_rank"] is None else candidate_result["reciprocal_rank"] - base_result["reciprocal_rank"],
            "ndcg_delta": None if base_result["ndcg"] is None else candidate_result["ndcg"] - base_result["ndcg"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if candidate_strategy == "hybrid_rerank":
        section("Reranking impact", "See which fused candidates the cross-encoder promoted or demoted.")
        movement_rows = []
        for result in candidate["results"]:
            for movement in result["reranking_trace"]:
                movement_rows.append({"question": result["question"], **movement})
        if movement_rows:
            movement_frame = pd.DataFrame(movement_rows).sort_values("movement", ascending=False)
            st.dataframe(movement_frame.head(30), use_container_width=True, hide_index=True)
        else:
            st.info("No reranked hits were returned for this benchmark.")
        stage_rows = []
        for label, report in (("baseline", baseline), ("candidate", candidate)):
            for stage, values in report["stage_latency_ms"].items():
                stage_rows.append({"pipeline": label, "stage": stage, **values})
        st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)

    artifact_path = save_comparison_artifact(baseline, candidate, comparison, SETTINGS.outputs_dir)
    st.caption(f"Saved experiment IDs: {baseline['experiment_id']} · {candidate['experiment_id']} · Comparison: {artifact_path}")

footer()
