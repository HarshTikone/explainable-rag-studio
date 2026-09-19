# Demo walkthrough script: public portfolio release

Record this only after the merged `main` SHA is live on Render and the final smoke test passes.
Target length: 3–5 minutes. Use the public URL, not a local build, so cold-start and fallback
behavior shown on camera match what a reviewer will experience.

## Before recording

1. Confirm Render reports the Free plan and no disk or paid add-ons.
2. Confirm the deployed commit is the approved merge SHA.
3. Open <https://explainable-rag-studio-demo.onrender.com> and allow up to roughly one minute for
   the free instance to wake.
4. Smoke-test Home, What is RAG, Ask & Explain, and Results.
5. Run all four sample questions and confirm their Groq/fallback badges are truthful.

## Script

### 0:00–0:35 — What the project demonstrates

> “Explainable RAG Studio is a reliability workbench, not a chatbot wrapper. This hosted profile
> runs on a free 512 MB instance, so it deliberately uses BM25 retrieval and exact cited evidence.
> The repository also retains a full hybrid, reranked, locally verified platform for engineering
> and evaluation.”

On Home, point out the hosted-profile label, sanitized-corpus status, cold-start notice, and the
absence of upload or administration controls.

### 0:35–1:15 — Explain the pipeline

Open **What is RAG**. Briefly show retrieval, answer selection, citations, and strict abstention.
Mention that Groq can select at most two verbatim evidence sentences; invalid output, timeout,
quota, or provider failure automatically falls back locally.

### 1:15–2:40 — Show answers and a hard negative

Open **Ask & Explain** and run these tested prompts:

1. `How long are Aegis audit events retained?` — show `400 days` and its citation.
2. `What incident ID investigated Meridian clock skew?` — show `ID-2026-014`.
3. `Are current Aegis audit events retained for 400 or 90 days?` — show that the current
   `400 days` evidence wins over the obsolete 90-day document.

For one response, expand the retrieval trace and exact generator context. Point out the visible
badge: **Groq-assisted**, **Exact extractive fallback**, or **Quota fallback**. The badge is an
operating fact, not a quality score.

### 2:40–3:15 — Show safe abstention

Run `What is the manufacturing cost of Northstar's hardware appliance?`. The corpus does not
contain that fact, so the correct answer is an abstention. Emphasize that unrelated Northstar
evidence is not converted into a confident answer.

### 3:15–4:15 — Show measured evidence honestly

Open **Results**. Show the 112-question, six-category benchmark and its 25 unanswerable questions
(22.3%). Explain the profile distinction:

- Hosted: BM25, exact evidence, no local embedding/reranking/NLI model.
- Full platform: `hybrid_rerank` plus pinned `cross-encoder/nli-deberta-v3-xsmall` verification.

Call out rejected gates as openly as promoted ones. Read the decision and measurements from the
committed Results artifact; do not quote the local pre-release diagnostic. Record only after
retrieval, grounding, runtime, and security are all promoted together for the deployed SHA.

### 4:15–4:35 — Close

> “Every displayed answer is traceable to evidence, every fallback is visible, and the project’s
> limitations are part of the product. The same repository contains the benchmark, release gates,
> security controls, and production-profile implementation behind this demo.”

## After recording

- Verify every number visible in the video against the committed release artifact.
- Link the finished recording from `README.md` only after the live URL and deployed SHA still pass
  the final smoke test.
