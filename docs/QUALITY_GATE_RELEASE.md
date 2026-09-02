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

The complete run builds an isolated 60-card FAISS index, compares hybrid RRF with reranking at `top_k=5`, locks the retained retrieval strategy, compares grounding policies from cached inputs, evaluates the sealed claim benchmark, and writes `outputs/releases/<release_id>/quality_gate.json`.

## Retained evidence

Each release directory contains the source, corpus, benchmark, retrieval, draft-claim, and grounding-policy fingerprints; hardware and dependency metadata; experiment reports; cached draft bundles; and separate retrieval, grounding, runtime, and overall decisions. The manual GitHub Actions workflow runs on Python 3.11, builds Docker, checks Streamlit and FastAPI health, uploads the complete directory, and generates the compact portfolio summary.

No performance claim should be copied into the roadmap or portfolio unless it appears in a validated retained artifact.

## Current retained result

Local release `20260901T230637Z` validated successfully as an artifact and closed every decision:

- Reranking was rejected despite improving Recall@5, MRR, and nDCG@5 because candidate retrieval p95 was 1980.75 ms.
- Grounding was rejected because held-out macro F1 was 0.6858, contradiction recall was 0.8125, answer regressions exceeded their budgets, and verification p95 was 1538.63 ms.
- Citation validity, displayed claim-citation coverage, unsupported-claim exposure, and answer coverage passed.
- Runtime was rejected because this host uses Python 3.12 and has no Docker installation; the manual Python 3.11 workflow is the authoritative remaining runtime check.

The generated machine-readable summary is `docs/benchmarks/quality-gate-reference.json`. No improvement is claimed.
