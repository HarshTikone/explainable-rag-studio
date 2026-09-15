# Advanced Build Plan: Explainable RAG Studio

Updated: 2026-08-30

## Product direction

Evolve the repository from a PDF question-answering demo into an **evaluation-first RAG reliability workbench**. The finished product should let an engineer construct retrieval variants, run a versioned benchmark, inspect failures at every stage, and promote a configuration only when it improves quality without violating latency, cost, or security budgets.

The differentiator is not another chat interface. It is the ability to answer: **Why did retrieval fail, which change fixed it, and what evidence proves the improvement?**

## Research basis

- Anthropic reports that lexical BM25 complements embeddings for exact terms, contextualized chunks improve retrieval, and reranking can further reduce failed retrievals. These techniques must be evaluated on this project's own corpus rather than adopted without measurement: [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval).
- RAG quality should be decomposed into answer correctness, answer relevance, groundedness, and retrieval relevance instead of represented by one keyword-accuracy number: [LangSmith RAG evaluation guide](https://docs.langchain.com/langsmith/evaluate-rag-tutorial).
- OWASP identifies unauthorized vector access, cross-context leakage, embedding inversion, poisoned sources, and missing retrieval audit logs as concrete RAG risks: [OWASP LLM08](https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/).
- Production telemetry should use consistent traces, metrics, and model attributes. OpenTelemetry provides shared semantic conventions for those signals: [OpenTelemetry semantic conventions](https://opentelemetry.io/docs/concepts/semantic-conventions/).

## Success metrics

The project is advanced only when it can publish reproducible numbers. Initial targets:

| Dimension | Target |
|---|---:|
| Curated benchmark | 100+ questions across at least 6 categories |
| Unanswerable/adversarial coverage | At least 20% of benchmark |
| Retrieval Recall@5 | At least 0.85 on labeled questions |
| Citation validity | 1.00 |
| Abstention accuracy | At least 0.90 |
| p95 end-to-end latency | Declared and enforced per deployment tier |
| Core test coverage | At least 80% |
| Cross-tenant leakage tests | 100% passing |

Targets are acceptance gates, not claims. Baseline measurements must be committed before reporting an improvement.

## Milestone 1 — Evaluation foundation (implemented in this update)

Duration: 1 week

- Version the evaluation schema.
- Label gold relevant chunk IDs and answerability.
- Measure hit rate, Recall@k, Precision@k, MRR, citation validity, deterministic answer accuracy, and abstention accuracy.
- Preserve `None` for unlabeled retrieval metrics so missing labels are never presented as poor performance.
- Add category labels for failure slicing.
- Add unit tests for metric correctness.
- Next: store pipeline configuration, model identifiers, corpus hash, Git commit, latency, and cost with every experiment.

Exit gate: a baseline report can be reproduced from a clean checkout and compared with a candidate report.

## Milestone 2 — Hybrid retrieval laboratory (baseline + hybrid implemented)

Duration: 2 weeks

- ✅ Add BM25 lexical retrieval beside FAISS dense retrieval.
- ✅ Combine independent rankings using Reciprocal Rank Fusion.
- Retrieve a wider candidate pool and apply a cross-encoder reranker.
- ✅ Expose dense and lexical ranks, scores, and fusion stages in the UI.
- ✅ Add a sanitized 30-question benchmark covering exact-term, identifier, paraphrase, multi-hop, and unanswerable categories.
- ✅ Compare dense, dense+MMR, and hybrid pipelines using identical corpus and benchmark fingerprints.
- Next: add cross-encoder reranking and include it as the fourth pipeline.

Exit gate: the winning pipeline improves labeled Recall@5 or MRR without exceeding the declared p95 latency budget.

## Milestone 3 — Context-aware ingestion (implemented)

Duration: 2 weeks

- ✅ Add DOCX, Markdown, HTML, scanned-PDF OCR, and table extraction.
- ✅ Introduce heading-aware and parent-child chunking.
- ✅ Generate deterministic and optional cached Gemini context using document and section metadata.
- ✅ Add source checksum, deduplication, versioning, incremental update, and delete propagation.
- ✅ Record parser, chunker, context, embedding, dependency, and schema versions in the index manifest.
- ✅ Move ingestion to SQLite-backed background jobs with visible progress and retry states.

Exit gate: an index can be rebuilt deterministically, and document updates do not leave stale searchable chunks.

## Milestone 4 — Claim-level grounding

Duration: 1–2 weeks

- ✅ Generate schema-constrained atomic claims with retrieved chunk IDs and an offline extractive fallback.
- ✅ Verify citations, identifiers, numbers, negation, entailment, and contradictions with a local CPU NLI model.
- ✅ Remove unsupported and disputed claims under an explicit strict abstention policy.
- ✅ Display claim-specific conflicts, evidence excerpts, and verification latency.
- ✅ Persist low-confidence and disputed cases in an append-only SQLite human-review workflow.
- ✅ Ran and retained the real CPU 96-case plus 60-question diagnostic as release `20260901T230637Z`; grounding was rejected, so no improvement is claimed.

Exit gate: unsupported claims are flagged before display and citation validity remains 100% in CI.

## Milestone 5 — Security and multi-tenancy (implemented)

Duration: 2 weeks

- ✅ Add users, organizations, corpus ownership, and permission-aware retrieval filters. (`backend/security_models.py` roles/scopes, `backend/tenant_store.py` physically isolated per-organization FAISS stores)
- ✅ Enforce access filtering inside retrieval rather than after retrieval. (`backend/retriever.py` + `security_models.RetrievalScope`, verified in `tests/test_tenant_isolation.py` across all 4 retrieval strategies)
- ✅ Validate files, MIME type, size, source trust, and extracted content. (`backend/security_scanner.py`)
- ✅ Detect instruction-like text and hidden-content anomalies during ingestion. (`backend/security_scanner.py` prompt-injection pattern detection, hidden-HTML/base64-payload detection, quarantine)
- ✅ Add red-team cases for indirect prompt injection, poisoning, data exfiltration, and tenant leakage. (`tests/test_security_scanner.py`, `tests/test_security.py`, `tests/test_api_security.py`, `tests/test_tenant_isolation.py`)
- ✅ Add immutable retrieval audit events with safe redaction. (`backend/security.py` / `backend/postgres_security.py`, hash-chained HMAC audit log with `verify_audit_chain`)
- ✅ Remove personal résumé PDFs and other private material from the public Git history. (confirmed absent via `git log --all --diff-filter=A`; history rewrite documented in `docs/security-history-cleanup.md`)

Exit gate: automated tests prove that no query can retrieve or cite another tenant's document. **Met** — `tests/test_tenant_isolation.py` passes for all 4 retrieval strategies.

## Milestone 6 — Production platform

Duration: 2 weeks

- ✅ Replace local-only metadata with PostgreSQL and pgvector behind a storage interface; keep FAISS for the lightweight local profile.
- ✅ Add async ingestion workers, idempotency keys, streaming answers, timeouts, retries, rate limits, authentication, and API versioning.
- ❌ Instrument parsing, embedding, retrieval, fusion, reranking, generation, and citation verification with OpenTelemetry. Not started — no `opentelemetry-*` dependency exists anywhere in this repo. What does exist is a homegrown, non-OTel latency-tracking system spread across `backend/eval.py`, `backend/experiments.py`, `backend/grounding.py`, `backend/retriever.py`, and `backend/qa.py` (p50/p95/p99 per stage, surfaced on the Streamlit "Latency Dashboard" page) — real, but not what this bullet asked for, and it doesn't cover the rest of the next bullet.
- ⚠️ Track p50/p95/p99 latency, token usage, cost, cache hit rate, errors, and quality feedback. Latency: done (see above, homegrown). Token usage, cost, and cache hit rate: not tracked anywhere in the codebase. Quality feedback: covered separately by the grounding-review reviewer-label pipeline (`docs/CLAIM_LEVEL_GROUNDING.md`).
- ✅ Add database migrations (Alembic, `deploy/postgres/`), backups (`scripts/platform_backup.py`/`restore_validate.py`, exercised in CI), restore documentation, load tests (`scripts/load_test.py`, new this session), and deployment runbooks (`docs/DEPLOYMENT_RUNBOOK.md`, new this session).

Exit gate: the container passes health checks (✅, CI-verified), load targets (⚠️ load test now exists and runs in CI, but is report-only — no promotion threshold has been set, since there's no prior baseline to set a defensible one against), migration tests (✅), and a documented recovery exercise (✅, `docs/DEPLOYMENT_RUNBOOK.md`, backed by real repeated CI verification, not just a written procedure). **Real OpenTelemetry instrumentation and token/cost/cache-hit tracking remain genuinely unimplemented** — this is the honest state, not a checklist formality.

## Milestone 7 — Portfolio evidence

Duration: 1 week

- ✅ Commit the architecture diagram (`docs/ARCHITECTURE.md`), threat model (`docs/THREAT_MODEL.md`),
  benchmark methodology (`docs/RERANKING_BENCHMARK.md`, `docs/CLAIM_LEVEL_GROUNDING.md`,
  `docs/QUALITY_GATE_RELEASE.md`), experiment table (the 6.5.1–6.5.2 results table above, and
  `docs/RERANKING_BENCHMARK.md`'s reference run record), and load-test report
  (`scripts/load_test.py`, wired into `release-quality.yml`, report-only for now — no prior
  baseline exists yet to set a defensible latency budget against).
- ✅ Write two engineering case studies: `docs/case-studies/retrieval-improvement.md` and
  `docs/case-studies/rag-security-testing.md`.
- ✅ Report only reproducible metrics generated from committed experiment manifests — every number
  in the items above traces to a retained CI artifact or a currently-committed doc, not a fresh
  claim.
- ⏳ Record a short failure-to-fix walkthrough using the same benchmark before and after hybrid
  retrieval. Script written (`docs/DEMO_WALKTHROUGH_SCRIPT.md`); the actual recording needs a
  human at a keyboard, not something this session can produce.
- ⏳ Publish a live sanitized demo and OpenAPI documentation. Blocked on a hosting decision and
  credentials — deliberately not guessed at. Owner will provide hosting target and access in a
  later session; FastAPI already serves OpenAPI docs live at `/docs`/`/openapi.json` once deployed,
  so this is a deploy-target problem, not a missing-artifact problem.

## Milestone 6.5 — Close the outstanding quality gates (attempted 2026-09-12)

Duration: 2–3 weeks. Genuinely unfinished per `docs/QUALITY_GATE_RELEASE.md`'s retained result: benchmark under target (60 questions/6 categories/16.7% adversarial vs. 100+/20%), reranking rejected on latency (candidate-retrieval p95 1980.75 ms vs. the 1500 ms budget in `backend/experiments.py`), grounding rejected on both quality (held-out macro F1 0.686, contradiction recall 0.8125) and latency (verification p95 1538.63 ms), and the runtime gate never run for real (prior attempts logged Python 3.12 with no Docker).

**This session's attempt hit a different, more specific blocker than the prior "no Docker" note**, worth recording precisely so the next attempt doesn't repeat the diagnosis:

- Python 3.11.15 and a working Docker Engine + Compose were both available here (`dockerd` started cleanly as root).
- However, this session's outbound network egress is allowlisted by organization policy, and the policy explicitly rejects (403, "policy denial", confirmed via the agent-proxy status endpoint — not a transient failure) three hosts that essentially all of Milestone 6.5's empirical work depends on:
  - `huggingface.co` — blocks downloading the embedding model (`sentence-transformers/all-MiniLM-L6-v2`), the reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`), and the grounding NLI model (`cross-encoder/nli-deberta-v3-xsmall`) used by `backend/embeddings.py`, `backend/onnx_cross_encoder.py`, and `scripts/prefetch_models.py`.
  - `openaipublic.blob.core.windows.net` — blocks `tiktoken`'s one-time download of the `cl100k_base` vocabulary used by token-based chunking (`backend/chunking.py`), which in turn breaks contextual ingestion for arbitrary documents (see 6.5.4 below for the one case where this was still worked around).
  - `production.cloudfront.docker.com` — blocks pulling image layers for the Compose stack's base images (Docker Hub's registry API answered, but blob/layer fetches from its CDN were rejected), so `docker compose build` cannot complete even though the daemon runs fine.
- Per this environment's own proxy guidance, a 403 policy denial is to be reported, not routed around — there is no legitimate workaround from inside the session (no alternate mirror was probed further once the policy-denial pattern was confirmed across independent hosts).
- Net effect: 91/106 fast tests pass with a from-scratch `pip install -r requirements.txt` under Python 3.11; the 13 failures (`test_contextual_ingestion.py`, `test_demo_benchmark.py::test_benchmark_labels_resolve_against_rebuilt_default_chunks`, `test_grounding_eval.py::test_public_grounding_benchmark_has_96_valid_stable_cases`, `test_release_quality.py::test_mocked_release_writes_and_validates_schema_33_artifact`) all trace to one of the three blocked hosts above, not to application logic.
- **What unblocks this milestone:** run it in an environment whose egress policy allows `huggingface.co`, `production.cloudfront.docker.com` (or an alternate Docker registry mirror), and `openaipublic.blob.core.windows.net` — or pre-provision the required model weights, ONNX files, tokenizer vocab, and base image layers into the environment before the session starts so no runtime download is needed.
- No code, config, or threshold was changed to compensate for this — per this project's own rule, unverified performance or quality claims are not written into this document or into `docs/benchmarks/quality-gate-reference.json`. The reranker and grounding latency/quality fixes described below in 6.5.2/6.5.3 remain to be attempted once real model access is available; `backend/reranker.py`'s batching and `backend/onnx_cross_encoder.py`'s dynamic per-request padding were read and found structurally sound on inspection, so the highest-leverage remaining lever is very likely candidate-pool size (`SETTINGS.rerank_candidates`, currently 30) per the original plan's own ranking — but that change should not ship without a live `scripts/run_release_validation.py` pass to confirm Recall@5/MRR are retained at a smaller pool.

### 6.5.4 — Benchmark expansion (done in this session, network-free)

Unlike 6.5.1–6.5.3, benchmark expansion does not require the models it was assumed to need. `backend/contextual_chunking.py`'s chunk-id hash (`sha256(document_id, heading_path, parent.ordinal, child_ordinal, text)`) does not depend on tiktoken's specific BPE algorithm — it only depends on tiktoken *not splitting* the chunk, which is guaranteed here because every `data/public_demo` document (200–530 bytes) sits far below the 420/1200-token child/parent limits, so `chunk_text_token_based`'s encode/decode round trip is a no-op and the hashed text is exactly the normalized heading plus paragraph. `document_id` (`backend/document_parsers.py:stable_document_id`) and the version-inference rule (`Status:` line, else filename suffix, else `"general"`) are pure-Python and also network-free. Reimplementing this formula from reading the source and testing it against all 29 chunk_ids already committed in `data/public_demo_benchmark.json` reproduced every one exactly, so it was used to compute correct chunk_ids for 12 previously-uncited documents (the 5 singleton docs 02–06 and the 8 access/response/routing/eval/storage/observe/locale/lifecycle current+incident docs whose facts had never been cited).

Result: `data/public_demo_benchmark.json` grew from 60 to 112 questions, all 6 categories still represented (17/18/17/17/18/25 for exact_term/identifier/paraphrase/hard_negative/multi_hop/unanswerable), unanswerable share up from 16.7% to 22.3% (target: 20%+). Every new item's answer was checked by hand against the cited document's text. `tests/test_demo_benchmark.py`'s schema test was loosened from exact counts (`== 60`, `== 10` per category) to floor checks (`>= 100` total, `>= 10` per category, `>= 20%` unanswerable) plus a new uniqueness check, since the project's own target is a floor, not an exact count. `test_benchmark_labels_resolve_against_rebuilt_default_chunks` (the test that actually rebuilds the corpus and checks `labeled_ids <= chunk_ids`) still needs tiktoken to run and so still fails in this sandbox, but it was not touched — the chunk-id and version-inference logic used to generate the new items *is* that test's logic, reimplemented and cross-checked, so it is expected to pass once run somewhere with network access.

### 6.5.1–6.5.2 — Runtime and reranker gates closed for real, via GitHub Actions (2026-09-12)

The sandbox network block described above is a property of this coding session's own container, not of every environment — `.github/workflows/release-quality.yml` already existed for exactly this reason (a manual `workflow_dispatch` job on a GitHub-hosted runner with normal internet access) but had never once completed: all three prior runs on `main` died inside "Build and start portable platform," and the retained result in `docs/QUALITY_GATE_RELEASE.md` explicitly says so ("this host uses Python 3.12 and has no Docker installation; the manual Python 3.11 workflow is the authoritative remaining runtime check").

Manually dispatching that workflow against this branch surfaced nine independent, genuine infrastructure bugs in sequence — each one only visible once the previous was fixed, since the pipeline died at the first failure every time:

1. `minio/minio` and `minio/mc` pulls denied by Docker Hub (anonymous pulls retired) → switched to `quay.io/minio/minio` and `quay.io/minio/mc`.
2. `scripts/generate_dev_secrets.py` wrote Docker secret files as `0600`; Compose bind-mounts them verbatim into containers running as a different, unprivileged UID → `0644`.
3. Keycloak's bootstrap-admin CLI validation runs before `_FILE`-suffixed secrets resolve, so `KC_BOOTSTRAP_ADMIN_PASSWORD_FILE` was invisible when checked → resolved both Keycloak secrets to plain env vars via a shell wrapper before invoking `kc.sh`.
4. `docker compose exec` doesn't inherit `start-service.sh`'s runtime-exported env vars (`DATABASE_URL`, etc.) → split that export logic into a shared, sourceable `deploy/env-secrets.sh`.
5. `python scripts/x.py` doesn't add `/app` to `sys.path` the way `uvicorn`/`alembic` do internally → `ENV PYTHONPATH=/app` in the Dockerfile.
6. `pg_dump`/`pg_restore` don't understand SQLAlchemy's `postgresql+psycopg://` scheme and silently fall back to a local unix socket instead of erroring → added `_libpq_url()` to strip the driver suffix.
7. `rag_app` (the app's own DB role) is deliberately `NOBYPASSRLS` for tenant isolation, so it cannot produce a complete `pg_dump`; Postgres correctly refuses rather than silently filtering rows → backup/restore now authenticate as `rag_owner` (the Postgres superuser) via a dedicated `ADMIN_DATABASE_URL`, mounted only into the `api` container.
8. MinIO requires a configured KMS backend to honor *any* `PutObject` server-side-encryption request (even plain AES256/SSE-S3), which this reference stack doesn't configure → dropped the redundant `ServerSideEncryption="AES256"` request, since the payload is already client-side envelope-encrypted (AES-256-GCM) before upload.
9. Debian's generic `postgresql-client` metapackage drifted ahead of the `pgvector/pgvector:pg16` server, so a newer `pg_restore` emitted session setup (`transaction_timeout`, added in PG17) the pg16 server rejects → pinned `postgresql-client-16` via the official PGDG apt repo.

Each fix was validated by re-dispatching the real workflow against GitHub's runner (not assumed) before moving to the next failure. The ninth run completed the full pipeline for the first time in this project's history, producing retained release `20260912T154030Z`:

| Decision | Result |
|---|---|
| Runtime | **promoted** — Python 3.11, Docker build, Streamlit health, API smoke, and dependency checks all passed |
| Retrieval | **promoted** — `hybrid_rerank` retained; Recall@5 +0.0345, MRR +0.0218, nDCG@5 +0.0356, all latency and citation gates passed |
| Grounding | **rejected** — `strict_safe_abstention`; macro F1 0.668 and contradiction recall 0.8125 still below their bars; verification latency itself now passes (294 ms, well under budget). The −0.214 answer-accuracy regression against held-out is not a new finding here — it closely matches the −0.2333 regression `docs/CLAIM_LEVEL_GROUNDING.md` already documented for the 2026-09-01 retained release, so it's a pre-existing, unresolved gap this run simply re-measured, not something this run introduced. |
| Security | **promoted** |
| Overall | **rejected** — blocked solely by the grounding decision |

This closes Milestone 6.5.1 (the runtime gate has now genuinely run and passed) and, incidentally, Milestone 6.5.2 as originally scoped (the reranker latency and quality gates both pass on this run) — no config or threshold was changed to get there; `SETTINGS.rerank_candidates` is untouched at 30. What remains of Milestone 6.5.3 is a real, separate ML problem, not an infrastructure one: recalibrating the grounding policy's premise strategy and thresholds via `scripts/calibrate_grounding.py --allow-small-fallback` against the 48 calibration cases only, with particular attention to contradiction recall (the weaker of the two failing metrics) and the newly-visible answer-accuracy regression.

## Recommended implementation order

1. Finish experiment manifests and baseline benchmark.
2. Implement BM25 and Reciprocal Rank Fusion.
3. Add reranking and compare four retrieval variants.
4. Add structured claim/citation verification.
5. Build context-aware ingestion and lifecycle management.
6. Add tenant-aware security and adversarial evaluation.
7. Move to shared storage and distributed observability.

Agentic workflows, Graph RAG, fine-tuning, and Kubernetes are intentionally deferred. They should be added only when a benchmarked use case requires them.
