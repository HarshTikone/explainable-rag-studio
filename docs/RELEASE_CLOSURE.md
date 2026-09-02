# Release closure runbook

## Retained decisions

The release workflow runs on Python 3.11 and treats the following as independent decisions:

- retrieval promotion;
- grounding promotion;
- production runtime readiness;
- adversarial security promotion;
- overall release readiness.

A failure retains `hybrid_rrf` and strict safe abstention. Benchmark labels, held-out cases, and thresholds are never changed merely to obtain a passing result.

## Required sequence

1. Run the model-free Python 3.11 suite and coverage workflow.
2. Calibrate only against the 48 calibration claim cases. The calibration function rejects any collection containing held-out cases.
3. Lock model revisions, ONNX file, thresholds, premise strategy, guards, batch size, and thread count.
4. Warm each model once, then run three measured passes against the frozen 60-question and 48-case held-out sets.
5. Build and start the full Compose platform and smoke-test `/ready`, `/ask`, document, review, quarantine, and audit routes.
6. Create an encrypted database backup and restore it into a distinct database.
7. Run dependency, static-security, PII, and full-history scans.
8. Produce immutable release and security artifacts. The workflow fails if either overall decision is not `promoted`.

## History rewrite

The authorized rewrite removes exactly:

- `data/Harsh_Resume_Intern (1).pdf`
- `data/Harsh_Resume_Intern.pdf`
- `data/Harsh_Tikone_Resume (20).pdf`

Before rewriting, fetch `origin/main`, record its exact object ID, freeze collaboration, and create a checksummed bare mirror outside the working repository. Rewrite all references with `git filter-repo`, scan paths and known blob hashes, then force-push only `main` with an exact force-with-lease value. Validate from a fresh clone and require collaborators to reclone. Keep the mirror offline for controlled recovery.

The release tag is created only after the fresh clone passes Python 3.11, Compose, restore, security, and ML promotion gates.
