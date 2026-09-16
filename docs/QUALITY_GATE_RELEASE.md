# Quality-Gate Release Procedure

## Purpose

This release track closes every pending retrieval, grounding, and runtime gate with a retained `promoted` or `rejected` decision. A failed candidate remains disabled; benchmark labels and held-out results are never used to force a pass.

## Evaluation rules

- The 48 grounding calibration cases are the only cases used to select premise strategy and thresholds.
- The 48 held-out cases are evaluated only after a policy is locked and fingerprinted.
- Answer coverage uses answerable questions only.
- Citation validity is aggregated over displayed citations; correct abstentions do not create fake invalid citations.
- `structured_unfiltered` and `strict_grounded` reuse byte-identical retrieval results and draft claims.
- Models are warmed once and three measured passes feed the latency percentiles.

## Commands

```bash
python scripts/calibrate_grounding.py --allow-small-fallback
python scripts/evaluate_grounding_policy.py outputs/grounding_policy.json
python scripts/run_release_validation.py
python scripts/run_release_validation.py --validate outputs/releases/<release_id>
```

The complete run builds an isolated 60-card FAISS index, evaluates the 112-question QA benchmark
(25 unanswerable; 22.3%), compares hybrid RRF with reranking at `top_k=5`, locks the retained
retrieval strategy, compares grounding policies from cached inputs, evaluates the sealed claim
benchmark, and writes `outputs/releases/<release_id>/quality_gate.json`.

## Retained evidence

Each release directory contains the source, corpus, benchmark, retrieval, draft-claim, and grounding-policy fingerprints; hardware and dependency metadata; experiment reports; cached draft bundles; and separate retrieval, grounding, runtime, and overall decisions. The manual GitHub Actions workflow runs on Python 3.11, builds Docker, checks Streamlit and FastAPI health, uploads the complete directory, and generates the compact portfolio summary.

No performance claim should be copied into the roadmap or portfolio unless it appears in a validated retained artifact.

## Current decision

The QA-remediation candidate's complete local diagnostic (`20260916T060948Z`) cleared both model
quality decisions without changing labels or gates: `hybrid_rerank` improved MRR by 0.0379 and
nDCG@5 by 0.0349 while staying inside the recall budget; grounding measured macro F1 0.8851,
supported precision 1.00, contradiction recall 1.00, zero unsupported exposure, and zero
answer/abstention regression. This is diagnostic evidence, not the retained release: the host used
Python 3.12 and could not execute Docker, so runtime remained rejected. Python 3.11 CI must rerun
the complete workflow, promote runtime/security, and produce the committed artifact before the
live service changes.

`docs/benchmarks/quality-gate-reference.json` is the committed machine-readable artifact rendered
by the Results page. A new release must regenerate and validate that artifact before any newer
numbers are called retained evidence; this document never turns a local diagnostic into a retained
promotion.
