# Case study: closing a retrieval quality gain that a diagnostic said would break latency

**tl;dr**: a hybrid-retrieval + cross-encoder-reranking strategy measured a real quality gain
(Recall@5 +0.034, MRR +0.022, nDCG@5 +0.036) but the first measurement also showed it blowing a
1.5-second latency budget by nearly 400ms. Rather than loosen the budget or ship the strategy
anyway, the project treated the gate as correct and went looking for why the measurement and the
real system disagreed. The answer was infrastructure, not the retrieval strategy — and re-running
the exact same comparison on real CI infrastructure showed a latency delta of 613ms, comfortably
inside budget.

## The setup

The retrieval laboratory (`docs/RERANKING_BENCHMARK.md`) compares four strategies —
`dense`, `dense_mmr`, `hybrid_rrf`, `hybrid_rerank` — on identical indexed chunks and a 60-question
labeled benchmark, and gates promotion on four conditions: matching corpus/config fingerprints, an
MRR-or-nDCG improvement of at least 0.03, no more than 0.01 Recall@5 regression, exact citation
validity, and candidate retrieval p95 under 1.5 seconds. `hybrid_rerank` fuses dense and BM25
retrieval with Reciprocal Rank Fusion, then reranks the top 30 fused candidates with
`cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU, ONNX-quantized, ~22.7M parameters).

## First measurement: quality real, latency over budget

The first recorded comparison (retained release `20260901T230637Z`, a dirty-tree Python 3.12 run
outside CI) showed the quality side of the hypothesis clearly confirmed:

| Metric | Delta |
|---|---|
| MRR | +0.058 |
| nDCG@5 | +0.063 |
| Recall@5 | +0.040 |

But candidate retrieval p95 came in at 1980.75ms — a +1896ms delta over the `hybrid_rrf` baseline,
and 480ms over the 1500ms budget. Gate result: rejected. The honest, disciplined move at that
point was *not* to relax the 1.5s budget to make the number pass — the budget represents a real
constraint (interactive query latency), and the whole point of a promotion gate is that it doesn't
bend to the result it's supposed to be checking.

## Finding out why, instead of assuming it was the model

The obvious hypothesis — "the reranker is just too slow" — didn't hold up on inspection: a 22.7M
parameter cross-encoder, CPU, ONNX int8, batch size 16, doing at most 30 pairs per query, is not
architecturally a 2-second operation. The more likely explanation was the measurement environment
itself: this run happened on a "dirty tree" (uncommitted local state) on Windows, Python 3.12 —
not the project's actual target runtime (Python 3.11, containerized), and not measured on
consistent, dedicated hardware.

`release-quality.yml` already existed as the real check — a `workflow_dispatch` job that builds
and runs the full Docker Compose stack on a GitHub Actions runner and re-measures the same
comparison for real. It had simply never completed: three prior attempts on `main` all died
inside "Build and start portable platform," and the project's own retained result openly recorded
why ("this host uses Python 3.12 and has no Docker installation; the manual Python 3.11 workflow
is the authoritative remaining runtime check" — see `docs/QUALITY_GATE_RELEASE.md`).

## Getting the real check to run at all

Manually dispatching that workflow surfaced nine independent infrastructure bugs, each only
visible once the previous one was fixed (the pipeline died at the first failure every time) —
everything from a Docker Hub anonymous-pull policy change, to Docker secret file permissions, to
a Postgres client/server version mismatch that only manifested during backup/restore validation.
None of them were retrieval-strategy bugs; all of them were "can this stack actually boot and run
end-to-end on a real runner" bugs. Full list in `docs/ADVANCED_BUILD_PLAN.md`'s Milestone 6.5.1
section.

## The real number

Once the pipeline actually completed (retained release `20260912T154030Z`), the comparison
re-ran on real Python 3.11, real Docker, a real GitHub Actions runner:

| Metric | Diagnostic (2026-09-01) | Real CI (2026-09-12) |
|---|---|---|
| Recall@5 delta | +0.040 | +0.0345 |
| MRR delta | +0.058 | +0.0218 |
| nDCG@5 delta | +0.063 | +0.0356 |
| Retrieval p95 delta | +1896ms | **+613ms** |
| Gate result | Rejected (latency) | **Promoted** |

The quality gain held — smaller in absolute terms on the real run (still comfortably over the
0.03 threshold), consistent with a cleaner, more controlled measurement environment replacing a
dirty local diagnostic. The latency figure changed by more than 3x. Re-running the identical
comparison again on a later, unrelated CI run reproduced matching deltas almost exactly
(Recall@5 +0.0345, MRR +0.0218, nDCG@5 +0.0356, p95 delta +613.48ms) — this wasn't a lucky single
measurement.

## What this actually demonstrates

Not "reranking works" — that was already true in the first measurement. The real lesson is about
where to spend skepticism: a failing latency gate on a dirty-tree, wrong-Python-version, no-Docker
diagnostic is evidence about *that measurement*, not necessarily about the strategy under test.
The fix wasn't tuning the reranker, and it wasn't loosening the gate — it was getting a trustworthy
measurement environment working at all, then trusting what it said. The nine infrastructure fixes
that took (`docs/ADVANCED_BUILD_PLAN.md`) had nothing to do with retrieval and everything to do
with making the CI claim actually verifiable rather than assumed.
