# Explainable RAG Studio

> **An end-to-end, recruiter-ready Retrieval-Augmented Generation (RAG) system with explainability, evaluation, and latency observability.**

This project demonstrates how to build a **production-style RAG Document Q&A system** using modern NLP techniques. It is designed not only to *work*, but to clearly **explain every step of the RAG pipeline** to recruiters, clients, or non-technical stakeholders through an interactive UI.

---

## 🎥 Demo Video

▶️ **Project Walkthrough (3–5 min):**
[Watch the walkthrough (3-5 min)](https://app.govideolink.com/videos/0DSg0V06vaOuG9vvYxpv/?utm_source=direct&utm_medium=invite_link)

This short demo walks through:

* What problem RAG solves
* PDF ingestion and FAISS indexing
* Ask & Explain (retrieval + citations)
* Embedding visualization (UMAP)
* Evaluation and latency dashboard

> 📌 *Tip for reviewers:* Watch this video first to understand the system end-to-end in minutes.

---

## 🚀 What This Project Does

* Upload PDF, scanned PDF, DOCX, Markdown, HTML, and TXT documents
* Preserve headings and tables with **parent-child contextual chunking**
* Convert chunks into **vector embeddings**
* Store and search them efficiently using **FAISS**
* Compare dense, dense+MMR, **hybrid BM25 + vector retrieval**, and local cross-encoder reranking
* Fuse lexical and semantic rankings using **Reciprocal Rank Fusion**
* Rerank the top 30 fused candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`
* Track stable document versions, duplicate checks, background ingestion jobs, retries, and soft deletion
* Generate atomic answer claims with Gemini or a deterministic extractive fallback
* Verify every claim locally with deterministic guards and a CPU NLI cross-encoder
* Remove unsupported or disputed claims before display and queue uncertain cases for human review
* Bind each displayed claim to one to three verified citations
* Visualize retrieval, embeddings, and similarity scores
* Evaluate system accuracy using a reproducible **JSON-based benchmark**
* Track **latency and performance metrics** for each query

This mirrors how real-world RAG systems are built and evaluated in industry.

---

## 🧠 Why RAG?

Large Language Models (LLMs) are powerful, but they **hallucinate** when asked about private or unseen data. RAG solves this by:

1. Retrieving relevant document chunks
2. Injecting only those chunks into the LLM prompt
3. Generating answers **grounded in sources**

This system enforces grounding and provides citations so answers are **verifiable and trustworthy**.

---

## 🏗️ System Architecture

```
PDF Documents
      │
      ▼
Document Loader (PDF → Text)
      │
      ▼
Token-based Chunking (300–500 tokens, overlap)
      │
      ▼
Embedding Model (SentenceTransformers)
      │
      ▼
FAISS Vector Index (Cosine Similarity)
      │
      ▼
Retriever (Top-K / MMR)
      │
      ▼
Prompt Construction (Context + Rules)
      │
      ▼
Structured Gemini / extractive claims
      │
      ▼
Local NLI + conflict verification
      │
      ▼
Supported claims + evidence + metrics
```

---

## 🖥️ User Interface (Streamlit)

The project includes a **multi-page interactive Streamlit app**:

### 1️⃣ What is RAG?

* Client-friendly explanation of LLMs and hallucinations
* Step-by-step overview of the RAG pipeline

### 2️⃣ Ingest & Index

* Upload structured documents or load the public benchmark corpus
* Configure parent/child chunking and optional cached Gemini context
* Inspect background-job stages, warnings, retries, versions, and duplicate outcomes
* Preview heading paths, tables, contextual prefixes, and stable chunk IDs
* Soft-delete documents with immediate dense and lexical index propagation

### 3️⃣ Ask & Explain

* Ask natural language questions
* View retrieved chunks and similarity scores
* See the exact context sent to the LLM
* Inspect accepted and removed claims, evidence scores, conflicts, and strict abstentions
* Answers contain only claim-bound verified citations

### 4️⃣ Embedding Explorer

* 2D visualization of chunk embeddings using **UMAP**
* Shows semantic clustering of document content

### 5️⃣ Evaluation

* Upload a JSON evaluation set
* Measure accuracy automatically
* Inspect failure cases

### 6️⃣ Latency Dashboard

* Track retrieval time, generation time, total latency
* View performance trends across queries

### 7️⃣ Grounding Review

* Resolve disputed and low-confidence claim/evidence cases
* Compare cited and conflicting passages
* Export append-only reviewer labels as JSONL

---

## 📊 Evaluation Methodology

> The advanced implementation roadmap is maintained in [docs/ADVANCED_BUILD_PLAN.md](docs/ADVANCED_BUILD_PLAN.md).

## Portable production profile

The repository now includes an opt-in production profile built around PostgreSQL row-level security, pgvector, Redis/RQ workers, envelope-encrypted S3 storage, generic OIDC, scoped API keys, quotas, and immutable release evidence. Streamlit becomes an API-only frontend in this mode; tenant identity is always derived from the authenticated credential.

See [docs/PRODUCTION_PLATFORM.md](docs/PRODUCTION_PLATFORM.md) for deployment and migration guidance and [docs/RELEASE_CLOSURE.md](docs/RELEASE_CLOSURE.md) for the quality, security, backup, and authorized history-rewrite procedure. The local reference stack starts with:

```bash
python scripts/generate_dev_secrets.py
docker compose build
docker compose up -d
```

`hybrid_rrf` and strict safe abstention remain the retained defaults until real Python 3.11 CPU measurements produce promoted reranking and grounding artifacts.

The system supports **reproducible evaluation** using a JSON file:

```json
[
  {"question": "What is the purpose of the document?", "expected": "purpose"},
  {"question": "What technology is used?", "expected": "faiss"}
]
```

The evaluator supports the original `expected` phrase format and an advanced schema with `reference_answer`, gold `relevant_chunk_ids`, `answerable`, and `category` fields. Each run now reports:

* Deterministic answer accuracy
* Retrieval hit rate, Recall@k, Recall@5, Precision@k, MRR, nDCG@k, and nDCG@5 when gold chunks are labeled
* Citation validity against retrieved chunks
* Correct abstention on unanswerable questions
* Displayed-claim support, claim citation coverage, answer coverage, conflicts, and verification p50/p95

Start with `data/eval_set.example.json` and replace its placeholder references with manually verified labels from your corpus.

For a complete reproducible demo, enable **Use bundled sanitized demo corpus** on the ingestion page, build the index with the default chunk settings, then upload `data/public_demo_benchmark.json` on the Evaluation page. Its 60 labeled questions map deterministically to 60 single-chunk knowledge cards. The original six-card corpus remains available under `data/public_demo_small/`.

The default Evaluation comparison is `hybrid_rrf` versus `hybrid_rerank`. It writes schema 3.3 manifests with retrieval and grounding policy fingerprints, source-tree and runtime metadata, per-question results, stage latency, category slices, and a portfolio comparison artifact. See [the reranking benchmark methodology](docs/RERANKING_BENCHMARK.md), [the claim-grounding methodology](docs/CLAIM_LEVEL_GROUNDING.md), and [the release quality procedure](docs/QUALITY_GATE_RELEASE.md). Neither reranking nor grounding is claimed as an improvement until a retained CPU run passes its promotion gate.

The original baseline accuracy remains:

```
accuracy = correct_answers / total_questions
```

Evaluation results are saved to disk and displayed in the UI.

---

## ⚡ Performance & Latency

For every query, the system logs:

* Dense, lexical, fusion, and reranking stage latency
* Generation latency (Gemini)
* Claim verification latency and accepted/rejected/conflict counts
* Total end-to-end latency

This allows comparison between:

* Baseline vs tuned retrieval
* Different Top-K values
* Dense, MMR, hybrid RRF, and reranked hybrid retrieval

---

## 🛠️ Tech Stack

**Backend / ML**

* Python
* FAISS (vector database)
* SentenceTransformers (embeddings)
* DeBERTa-v3 xsmall NLI cross-encoder (local claim verification)
* MiniLM cross-encoder (local reranking)
* Gemini API (LLM)

**Frontend**

* Streamlit
* Plotly (visualizations)

**Evaluation & Ops**

* JSON-based benchmarks
* SQLite logging
* Latency tracking

---

## 📁 Project Structure

```
rag-studio/
│
├── app/                # Streamlit UI
│   ├── Home.py
│   └── pages/
│       ├── 1_What_is_RAG.py
│       ├── 2_Ingest_and_Index.py
│       ├── 3_Ask_and_Explain.py
│       ├── 4_Embedding_Explorer.py
│       ├── 5_Evaluation.py
│       └── 6_Latency_Dashboard.py
│
├── backend/            # Core RAG logic
│   ├── loaders.py
│   ├── chunking.py
│   ├── document_parsers.py
│   ├── contextual_chunking.py
│   ├── ingestion_registry.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── qa.py
│   ├── eval.py
│   └── experiments.py
│
├── data/               # Input PDFs
├── index/              # FAISS index (gitignored)
├── outputs/            # Logs & evaluation reports
├── requirements.txt
├── README.md
└── .env.example
```

---

## ▶️ How to Run Locally

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Add API key
cp .env.example .env
# Add GEMINI_API_KEY=your_key_here

# Run app
streamlit run app/Home.py
```

## Deploy

### Docker (recommended)

```bash
docker build -t explainable-rag-studio .
docker run --rm -p 8501:8501 -e GEMINI_API_KEY=your_key explainable-rag-studio
```

The image installs Tesseract OCR and prefetches the default reranker so deployed instances do not download it on their first reranked query. The container exposes the app on port `8501` and includes a Streamlit health check. On platforms such as Render, Railway, Fly.io, or Cloud Run, deploy the included `Dockerfile`, set `GEMINI_API_KEY` as a secret, and allow the platform to provide `PORT`.

Release validation can opt into the real-model smoke test after the model is cached:

```bash
RUN_RERANKER_SMOKE=1 pytest -q -m slow tests/test_reranker_slow.py
```

The complete offline release evidence run is headless:

```bash
python scripts/calibrate_grounding.py --allow-small-fallback
python scripts/evaluate_grounding_policy.py outputs/grounding_policy.json
python scripts/run_release_validation.py
python scripts/run_release_validation.py --validate outputs/releases/<release_id>
```

### Streamlit Community Cloud

Select `app/Home.py` as the entry point and add `GEMINI_API_KEY` in the app's secret settings. The checked-in `.streamlit/config.toml` supplies the production theme and server configuration.

> Uploaded documents, the FAISS index, and telemetry are currently stored on the local filesystem. Use a persistent volume for a single hosted instance; a multi-instance deployment should move documents, index metadata, and telemetry to shared managed storage.

---

## 🔐 Security & Best Practices

* `.env` is gitignored (API keys never committed)
* FAISS index is built locally (not stored in repo)
* System gracefully falls back to extractive mode if LLM key is missing
* Context generation falls back to deterministic document metadata if Gemini is missing or unavailable
* Index generations are validated in staging and atomically activated

---

## 💼 Why This Project Matters

This project demonstrates:

* Deep understanding of **RAG architectures**
* Strong **ML engineering discipline** (evaluation, latency, explainability)
* Ability to **explain complex systems clearly**
* Production-minded design with graceful fallbacks

It is intentionally built to be **interview-demo ready**.

---

## 📌 Future Improvements

* Calibrated domain-specific claim-verifier training
* LLM-as-judge evaluation
* Shared storage for multi-instance deployment

---

**Author:** Harsh Mahesh Tikone
**Focus:** AI / ML Engineering, RAG Systems, Applied LLMs
## Protected deployment

Production-style local deployment is deny-by-default. Set `SECURITY_MODE=required`, generate independent high-entropy values for `API_KEY_PEPPER` and `AUDIT_HMAC_KEY`, and keep both outside source control. Create the first organization and one-time owner key with:

```powershell
python scripts/bootstrap_security.py --organization "Example" --email "owner@example.com"
```

The secret is shown once. Indexes, lifecycle jobs, reviews, and caches live under `index/organizations/<organization_id>/`. The bearer key determines the organization; request bodies cannot select or override it.

For a read-only public demonstration, set `SECURITY_MODE=demo`. Anonymous access is limited to the sanitized `org_public` query experience and health checks. Uploads, document inventory, evaluation, review, telemetry, and administration still require a scoped key.

Security utilities:

- `python scripts/audit_security.py verify` verifies the append-only audit hash chain.
- `python scripts/migrate_tenants.py` migrates a provably public legacy index into `org_public`.
- `python scripts/scan_private_history.py` reports manifest-listed private files still reachable in Git history.

See [docs/security-history-cleanup.md](docs/security-history-cleanup.md) before any history rewrite. The rewrite is intentionally not automated.
