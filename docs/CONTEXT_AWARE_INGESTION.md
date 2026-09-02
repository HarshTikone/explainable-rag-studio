# Context-Aware Ingestion and Document Lifecycle

The context-aware profile replaces full synchronous rebuilds with durable background jobs, stable identities, version tracking, contextual child chunks, cached embeddings, and atomic FAISS activation. Legacy `faiss.index` plus `meta.json` indexes remain queryable; lifecycle operations create manifest schema `1.0`.

## Supported inputs

| Format | Structure | Tables | OCR |
|---|---|---|---|
| PDF | Page-aware text through PyMuPDF | pdfplumber | Tesseract for low-text pages |
| DOCX | Heading styles and paragraphs | Native Word tables | Not applicable |
| HTML | Headings, paragraphs, lists | HTML tables | Not applicable |
| Markdown | Headings and front matter | Markdown table blocks | Not applicable |
| TXT | Document body | No | Not applicable |

OCR is an optional runtime capability outside Docker. If a PDF contains both text and unreadable scanned pages, searchable content is retained with warnings. A scanned-only document fails with `OCR_REQUIRED` when Tesseract is unavailable.

## Data and identity

- `document_id` is derived from normalized logical filename identity.
- `document_version_id` is derived from the source SHA-256.
- `parent_id` represents one bounded structural section.
- `chunk_id` combines document identity, heading path, section/child ordinal, and normalized child text.
- `content_fingerprint` combines clean child text and its contextual prefix and keys the embedding cache.

The clean child remains in `text` for display and citation. `retrieval_text` prepends deterministic or cached Gemini context for dense and BM25 retrieval. `generation_text` adds a bounded parent excerpt for grounded answer generation.

The public benchmark now uses stable `chk_*` identifiers. `data/public_demo_chunk_aliases.json` maps every previous `c000001` identifier in both directions so old reports remain interpretable. Existing reports remain readable but their corpus fingerprints intentionally do not match the context-aware corpus.

## Lifecycle behavior

Jobs progress through validation, parsing, OCR/table extraction, chunking, context, embedding, index build, validation, and activation. SQLite uses WAL mode and leased jobs; expired running jobs return to the queue on worker startup.

- Identical checksum: `unchanged`; no parsing or embedding.
- Updated logical source: new active version, obsolete chunks deactivated, unchanged cached embeddings reused.
- New embedding model: all active chunks missing that model are embedded before activation.
- Soft deletion: the document, versions, and chunks become inactive and a new index generation excludes them.
- Activation failure: the staged index is discarded, the previous index remains queryable, and document activation state is restored.

Index activation writes FAISS, metadata, and the manifest into a sibling staging directory, verifies vector/metadata cardinality, then swaps the directory atomically. The manifest records parser, chunker, context, embedding, corpus, dependency, document-version, and capability information.

## API

- `POST /ingestion/jobs`: multipart uploads plus chunk/context options; returns one durable job ID per document.
- `GET /ingestion/jobs/{job_id}`: job state, progress, warnings, structured error, result, and events.
- `POST /ingestion/jobs/{job_id}/retry`: retry a failed job within the configured attempt limit.
- `POST /ingestion/jobs/{job_id}/cancel`: cancel a job that has not yet been claimed.
- `GET /documents`: active and historical document records with current versions and chunk counts.
- `DELETE /documents/{document_id}`: soft-delete and propagate removal to the index.
- `GET /health`: index profile and parser/OCR capability state.

The local API has no authentication; do not expose ingestion or deletion endpoints publicly until the security and multi-tenancy sprint is complete.

## Defaults and operations

- Child target: 420 tokens; overlap: 80; parent target: 1,200.
- Maximum file: 25 MB; maximum API batch: 10 documents.
- Context mode: deterministic. Gemini enhancement is optional, capped at 60 words, cached by model/prompt/content, and non-blocking on failure.
- Worker lease: 120 seconds; retry limit: 3 attempts.
- Registry: `outputs/ingestion.db`; managed uploads: `outputs/uploads/`; both require persistent storage in deployment.

Run `scripts/migrate_public_benchmark_ids.py` after intentionally changing chunk identity rules. The migration is deterministic and idempotent.
