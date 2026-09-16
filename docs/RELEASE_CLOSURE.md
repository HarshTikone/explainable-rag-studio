# Release closure runbook

## Retained decisions

The release workflow runs on Python 3.11 and treats the following as independent decisions:

- retrieval promotion;
- grounding promotion;
- production runtime readiness;
- adversarial security promotion;
- overall release readiness.

A failed grounding candidate retains strict safe abstention. The full production profile keeps its separately promoted `hybrid_rerank` retrieval strategy; the public 512 MB profile uses BM25 only. Benchmark labels, held-out cases, and thresholds are never changed merely to obtain a passing result.

## Required sequence

1. Run the model-free Python 3.11 suite and coverage workflow.
2. Calibrate only against the 48 calibration claim cases. The calibration function rejects any collection containing held-out cases.
3. Lock model revisions, ONNX file, thresholds, premise strategy, guards, batch size, and thread count.
4. Warm each model once, then run three measured passes against the frozen 112-question and 48-case held-out sets. The QA benchmark contains 25 unanswerable questions (22.3%).
5. Build and start the full Compose platform and smoke-test `/ready`, `/ask`, document, review, quarantine, and audit routes.
6. Create an encrypted database backup and restore it into a distinct database.
7. Run dependency, static-security, PII, and full-history scans.
8. Build the public image with no model prefetch, run `/ask` and all four public pages under a 512 MB limit, and require peak RSS below 450 MB.
9. Produce immutable release and security artifacts. The workflow fails if either overall decision is not `promoted`.

No live Render deployment occurs during this sequence. After a pull request passes and is merged,
deploy the exact merge SHA once; first verify the Render service still reports the Free plan and no
disk or paid add-ons.

## History rewrite

The authorized rewrite removes exactly:

- `data/Harsh_Resume_Intern (1).pdf`
- `data/Harsh_Resume_Intern.pdf`
- `data/Harsh_Tikone_Resume (20).pdf`

Before rewriting, fetch `origin/main`, record its exact object ID, freeze collaboration, and create a checksummed bare mirror outside the working repository. Rewrite all references with `git filter-repo`, scan paths and known blob hashes, then force-push only `main` with an exact force-with-lease value. Validate from a fresh clone and require collaborators to reclone. Keep the mirror offline for controlled recovery.

The release tag is created only after the fresh clone passes Python 3.11, Compose, restore, security, and ML promotion gates.
