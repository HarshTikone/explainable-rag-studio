# Cross-Encoder Reranking Benchmark

Status: measured and rejected on the retained local diagnostic; Python 3.11 release validation remains available through the manual workflow.

This laboratory compares `dense`, `dense_mmr`, `hybrid_rrf`, and `hybrid_rerank` on identical indexed chunks and questions. It is designed to decide whether the ranking gain from a cross-encoder justifies CPU latency. The project does not claim that reranking improves quality until a saved schema 3.3 comparison passes every promotion gate.

## Pipeline under test

`hybrid_rerank` independently retrieves up to `max(50, top_k × 5)` dense and BM25 candidates, fuses them with Reciprocal Rank Fusion (`rrf_k=60`), and scores the first 30 fused candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`. The final `top_k` is selected by cross-encoder score. Equal scores retain the fused order.

The reranker is CPU-only, lazy-loaded, uses batches of 16, and limits each query-document pair to 512 tokens. The selected model contains approximately 22.7 million parameters; its primary safetensors file is about 90.9 MB. Source: [official Hugging Face model card and files](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

## Benchmark design

- Corpus: 60 sanitized Markdown knowledge cards in `data/public_demo/`.
- Determinism: default chunk settings produce exactly one chunk per card and IDs `c000001` through `c000060`.
- Questions: 60 labeled examples, with 10 each for exact-term, identifier, paraphrase, hard-negative, multi-hop, and unanswerable categories.
- Hard negatives: overlapping current and retired procedures, similar incidents, exact identifiers, and version-specific facts.
- Labels: relevant chunk IDs, reference answers, answerability, and expected source versions.
- Generator: optional. Retrieval evaluation runs with the deterministic extractive fallback and does not require a Gemini key.

## Reproduction procedure

1. Use Python 3.11 and install `requirements-dev.txt`.
2. In Streamlit, enable the bundled sanitized corpus and build the index with 420-token chunks and 80-token overlap.
3. Open Evaluation and upload `data/public_demo_benchmark.json`.
4. Keep `top_k=5` for the release gate, select `hybrid_rrf` as baseline and `hybrid_rerank` as candidate, then run both.
5. Preserve the generated `outputs/experiments/<experiment_id>/` directories and `outputs/comparisons/*.json` artifact.
6. Record the CPU, logical core count, RAM, operating system, cold/warm model state, and whether power management was enabled.

## Metrics and promotion gate

The artifact includes aggregate and per-category Recall@k, Recall@5, Precision@k, hit rate, MRR, binary nDCG@k, nDCG@5, citation validity, per-question ranks, and p50/p95 stage latency.

Promotion requires all of the following:

- Matching corpus and benchmark fingerprints plus matching embedding, chunking, `top_k`, and dependency configuration.
- MRR or nDCG@5 improvement of at least 0.03.
- Recall@5 regression no worse than 0.01.
- Citation validity of exactly 1.00.
- Candidate retrieval p95 below 1.5 seconds on the documented CPU run.

## Reference run record

Populate this table only from a retained experiment artifact. Blank values are intentional; no performance claim has been measured in this checkout.

| Field | Value |
|---|---|
| Hardware | Windows 11, Intel Family 6 Model 186, 16 logical cores, Python 3.12.13 |
| Baseline experiment | `20260901T230734+0000-hybrid_rrf-761624dfbd5a` |
| Candidate experiment | `20260901T230748+0000-hybrid_rerank-7d6c47984341` |
| MRR delta | +0.0580 |
| nDCG@5 delta | +0.0629 |
| Recall@5 delta | +0.0400 |
| Retrieval p95 delta | +1896.21 ms |
| Candidate retrieval p95 | 1980.75 ms |
| Promotion result | Rejected: quality passed, 1.5-second latency gate failed |

These values come from retained release `20260901T230637Z`. They are a dirty-tree Python 3.12 diagnostic, so the runtime release gate remains rejected even though the retrieval decision is closed.

## Known limitations

- Labels are deterministic and reviewed in-repository, but the corpus is synthetic rather than production traffic.
- Binary relevance does not express graded usefulness among multiple supporting chunks.
- Cross-encoder scores are not calibrated probabilities and should only order candidates for the same query.
- CPU results vary with hardware, threading, cache warmth, and installed PyTorch builds.
- The benchmark is English-only and intentionally small; it does not prove domain-wide or multilingual performance.
- The Docker image is larger because it includes PyTorch, SentenceTransformers, and a prefetched model.
