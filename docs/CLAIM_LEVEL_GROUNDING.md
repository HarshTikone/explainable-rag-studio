# Claim-Level Grounding Laboratory

## Purpose

The grounding layer treats citations as verifiable relationships rather than answer decorations. The generator produces atomic claims with retrieved chunk IDs, a local NLI verifier scores each claim against the exact `generation_text` evidence, and strict rendering removes unsupported or disputed claims before users see them.

The implementation is complete, but no quality improvement is claimed until a real CPU run passes the promotion gates. The local verifier is an English NLI classifier, not a proof system; human review remains necessary for borderline, domain-specific, temporal, and multi-evidence cases.

## Pipeline

1. Retrieve evidence with any existing retrieval strategy.
2. Ask Gemini for schema-constrained atomic claims, or copy deterministic extractive sentences when Gemini is unavailable.
3. Reject citations that were not retrieved and run number, identifier, version, and negation checks.
4. Score claim/evidence pairs with `cross-encoder/nli-deberta-v3-xsmall` on CPU.
5. Score three bounded premises per evidence item: the best atomic sentence, its adjacent
   two-sentence window, and the bounded full chunk. Aggregate maximum entailment and contradiction
   independently.
6. Allow calibrated identifier/number-anchor or embedding-similarity support only when no hard
   conflict exists and the relevant lexical/NLI constraints pass.
7. Display only supported claims; abstain when none remain.
8. Persist disputed and low-confidence cases in `outputs/reviews.db` for local human review.

The verifier uses `generation_text`, including the bounded parent excerpt, because that is the evidence supplied to answer generation. It never verifies against `retrieval_text`, whose contextual prefix may be synthetic.

## Contracts and configuration

`POST /ask` retains `answer`, `citations`, `retrieved`, and `latency_ms`, and adds a `grounding` trace. Citation entries include their claim IDs, evidence excerpt, document/version IDs, heading path, page range, and verification score.

The strict policy is configured with:

- `GROUNDING_MODEL` and the pinned `GROUNDING_MODEL_REVISION=a150876415327c80daeff35ca6f68f5ed8cf5c24`
- `GROUNDING_BATCH_SIZE=16` and `GROUNDING_MAX_LENGTH=512`
- `GROUNDING_EVIDENCE_SCAN_K=8` and `GROUNDING_MAX_CLAIMS=8`
- `GROUNDING_ENTAILMENT_THRESHOLD=0.75`
- `GROUNDING_CONTRADICTION_THRESHOLD=0.70`
- `GROUNDING_LOW_CONFIDENCE_MARGIN=0.10`
- `GROUNDING_ANCHOR_OVERLAP_THRESHOLD` selected from `0.45, 0.55, 0.65, 0.75`
- `GROUNDING_SEMANTIC_SUPPORT_THRESHOLD` selected from `0.65, 0.70, 0.75, 0.80, 0.85`
- premise version `multi-window-v1`
- `GROUNDING_PROMPT_VERSION=1.0`

Review endpoints list cases, read their evidence and decision history, and submit `supported`, `unsupported`, or `contradicted` labels. Decisions are append-only and never change live thresholds automatically. These endpoints remain local-profile features until authentication is implemented.

## Reproducible evaluation

`data/grounding_benchmark.json` contains 96 reviewed claim/evidence cases: 48 calibration and 48 held-out. Six categories cover exact support, paraphrases, identifiers and numeric details, unsupported statements, hard contradictions, and temporal/negation conflicts. `scripts/build_grounding_benchmark.py` deterministically rebuilds the labels against the stable 60-card corpus. The separate end-to-end QA benchmark contains 112 questions, including 25 unanswerable cases (22.3%).

Experiment schema 3.3 records the verifier model/revision, locked policy and source-tree fingerprints, premise and guard versions, thresholds, prompt version, dependency versions, claim results, and verification latency. The evaluator reports displayed-claim support across displayed claims, citation coverage across claims, and answer coverage over answerable questions only.

Promotion requires held-out supported precision of at least 0.95, macro F1 of at least 0.85, contradiction recall of at least 0.85, displayed-claim support and citation coverage of 1.00, answer coverage of at least 0.85, and verification p95 below 1.5 seconds on the documented CPU reference run.

### Previous diagnostic result

The 31 August 2026 local Windows/Python 3.12 diagnostic processed all 96 pairs with the original single-sentence policy. Held-out supported precision was 1.00 and contradiction recall was 0.9375, but held-out macro F1 was 0.7301, below the 0.85 gate. This remains historical diagnostic evidence, not a promotion result. New reference runs must use the locked calibration workflow and schema 3.3 release artifact described in `QUALITY_GATE_RELEASE.md`.

### Current decision

The QA-remediation candidate's complete local diagnostic (`20260916T060948Z`) passed every
grounding gate: held-out macro F1 0.8851, supported precision 1.00, contradiction recall 1.00,
zero unsupported exposure, zero answer-accuracy regression, zero abstention regression, and
verification p95 627.5 ms. Thresholds and held-out labels were unchanged. Because that host used
Python 3.12 and could not run Docker, these are pre-release diagnostic results rather than retained
CI evidence; strict safe abstention remains the fail-closed behavior until the Python 3.11 release
workflow produces a fully promoted committed artifact.

## Release validation

Normal CI uses deterministic fake verifier scores and does not download the model. Release validation additionally runs the slow real-model smoke test, the 96-case claim benchmark, the 112-question QA benchmark, all four public Streamlit pages, a real deterministic `/ask`, and the Python 3.11 Docker build under the public 512 MB limit. Store the resulting artifact under `outputs/grounding/`; do not promote the milestone based on an unretained terminal run.

The hosted low-memory profile never loads this NLI model. Gemini may select at most two verbatim
cited evidence sentences, and exact-substring/citation guards either accept the complete draft or
discard it in favor of deterministic extraction.
