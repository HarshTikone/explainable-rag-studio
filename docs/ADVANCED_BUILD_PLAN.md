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

- ✅ Add users, organizations, corpus ownership, and permission-aware retrieval filters (`backend/tenant_store.py`, `backend/security_models.RetrievalScope`).
- ✅ Enforce access filtering inside retrieval rather than after retrieval — verified for all four retrieval strategies in `tests/test_tenant_isolation.py`.
- ✅ Validate files, MIME type, size, source trust, and extracted content (`backend/security_scanner.py`).
- ✅ Detect instruction-like text and hidden-content anomalies during ingestion (prompt-override, system-prompt-extraction, credential-exfiltration, and tool-instruction patterns, plus hidden-HTML and base64-payload detection in `backend/security_scanner.py`).
- ✅ Add red-team cases for indirect prompt injection, poisoning, data exfiltration, and tenant leakage (`tests/test_security_scanner.py`, `tests/test_tenant_isolation.py`, `tests/test_api_security.py`).
- ✅ Add immutable retrieval audit events with safe redaction — hash-chained, HMAC-signed audit log with `verify_audit_chain` (`backend/security.py`, `backend/postgres_security.py`).
- ✅ Remove personal résumé PDFs and other private material from the public Git history — confirmed absent from `git log --all --diff-filter=A` history as of 2026-09-11/12; see `docs/security-history-cleanup.md`.

Exit gate met: automated tests prove that no query can retrieve or cite another tenant's document (`tests/test_tenant_isolation.py`).

## Milestone 6 — Production platform

Duration: 2 weeks

- Replace local-only metadata with PostgreSQL and pgvector behind a storage interface; keep FAISS for the lightweight local profile.
- Add async ingestion workers, idempotency keys, streaming answers, timeouts, retries, rate limits, authentication, and API versioning.
- Instrument parsing, embedding, retrieval, fusion, reranking, generation, and citation verification with OpenTelemetry.
- Track p50/p95/p99 latency, token usage, cost, cache hit rate, errors, and quality feedback.
- Add database migrations, backups, restore documentation, load tests, and deployment runbooks.

Exit gate: the container passes health checks, load targets, migration tests, and a documented recovery exercise.

## Milestone 7 — Portfolio evidence

Duration: 1 week

- Publish a live sanitized demo and OpenAPI documentation.
- Record a short failure-to-fix walkthrough using the same benchmark before and after hybrid retrieval.
- Commit the architecture diagram, threat model, benchmark methodology, experiment table, and load-test report.
- Write two engineering case studies: retrieval improvement and RAG security testing.
- Report only reproducible metrics generated from committed experiment manifests.

## Recommended implementation order

1. Finish experiment manifests and baseline benchmark.
2. Implement BM25 and Reciprocal Rank Fusion.
3. Add reranking and compare four retrieval variants.
4. Add structured claim/citation verification.
5. Build context-aware ingestion and lifecycle management.
6. Add tenant-aware security and adversarial evaluation.
7. Move to shared storage and distributed observability.

Agentic workflows, Graph RAG, fine-tuning, and Kubernetes are intentionally deferred. They should be added only when a benchmarked use case requires them.
