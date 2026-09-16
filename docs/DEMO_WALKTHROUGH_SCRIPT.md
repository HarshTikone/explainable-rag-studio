# Demo walkthrough script: before/after hybrid retrieval

Milestone 7 (`docs/ADVANCED_BUILD_PLAN.md`) calls for "a short failure-to-fix walkthrough using
the same benchmark before and after hybrid retrieval." This is a script for recording that,
written so it can be followed step by step — it isn't itself the recording (this document can't
produce video), but everything in it maps to a real page/action in the app, so recording it is
mechanical rather than something that needs re-deriving on the day. Target length: 3–5 minutes.
Numbers to say on camera are the real, committed ones from `docs/RERANKING_BENCHMARK.md` — don't
substitute different numbers from a local run without updating both this script and that doc.

## Setup (before recording)

```bash
python scripts/generate_dev_secrets.py   # legacy/local profile is fine for this; no need for the full compose stack
streamlit run app/Home.py
```

Load the bundled sanitized corpus (`data/public_demo/`) and build the index with the release
defaults: 420-token chunks, 80-token overlap. Confirm the demo benchmark file
(`data/public_demo_benchmark.json`) is the one loaded on the Evaluation page — this keeps the
on-screen numbers reproducible by a viewer who clones the repo.

## Script

**0:00–0:30 — The problem, stated plainly**

> "This is a RAG system with 60 real Recall/MRR/nDCG-labeled questions against a 60-document
> corpus. The baseline here is `hybrid_rrf` — dense embeddings plus BM25 keyword search, fused
> with Reciprocal Rank Fusion. It works, but let's see where it actually fails."

Open the **Evaluation** page. Run `hybrid_rrf` against the benchmark. Let the aggregate metrics
render.

**0:30–1:30 — Show a real failure case, not a statistic**

> "Aggregate numbers hide the interesting part. Let's look at one failure."

Open a hard-negative or paraphrase-category question from the results (the benchmark has 10 of
each — pick one where `hybrid_rrf` ranks the correct chunk below top-3, visible in the per-question
ranks in the evaluation artifact). Switch to **Ask & Explain** and re-run that exact question.
Show the retrieved chunks and their similarity scores — narrate why a keyword/dense hybrid still
gets fooled here (e.g., two documents that share most of the same vocabulary but describe
different, sometimes contradictory, procedures — that's exactly what the hard-negative category is
designed to expose).

**1:30–2:30 — Apply the fix, re-run the same benchmark**

> "Now let's add cross-encoder reranking on top of the same fused candidates — same corpus, same
> questions, same fingerprint, only the ranking strategy changes."

Back on **Evaluation**, run `hybrid_rerank` (top 30 fused candidates re-scored by
`cross-encoder/ms-marco-MiniLM-L-6-v2`) against the identical benchmark. Show the same
previously-wrong question now ranking the correct chunk higher on **Ask & Explain**.

**2:30–3:30 — The numbers, and the part that almost didn't ship**

> "On the real CI measurement — Python 3.11, the full Docker stack, a GitHub Actions runner, not
> a local diagnostic — this was Recall@5 up 0.0345, MRR up 0.0218, nDCG@5 up 0.0356, and candidate
> retrieval p95 latency at 613 milliseconds, comfortably under the 1.5-second budget."

Show `docs/RERANKING_BENCHMARK.md`'s reference-run table on screen briefly. Optionally, mention
the actual failure-to-fix story behind *that* number: the very first measurement of this same
comparison showed the identical quality gain but a p95 of 1980ms — over budget — on a dirty-tree,
wrong-Python-version, no-Docker diagnostic. The fix wasn't the retrieval strategy; it was getting
a trustworthy CI measurement running at all (full story: `docs/case-studies/retrieval-improvement.md`).

**3:30–4:00 — Close**

> "Same corpus, same 60 questions, same fingerprint-verified inputs — only the ranking strategy
> changed, and it's the promotion gate itself, not a subjective read, that decided this shipped."

## After recording

- Trim to the target length; keep the on-screen benchmark numbers unedited/unobscured so they're
  independently checkable against `docs/RERANKING_BENCHMARK.md`.
- Publish alongside the live demo (Milestone 7's other unclosed item) rather than as a standalone
  asset with no way to try the thing being shown.
- Link it from `README.md`'s demo section once recorded.
