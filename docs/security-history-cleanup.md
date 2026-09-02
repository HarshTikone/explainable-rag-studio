# Private-history cleanup runbook

The working tree no longer uses the three résumé PDFs, but they remain reachable in Git history. The exact targets are recorded in `security/history-rewrite-manifest.json`.

No history rewrite is performed automatically. Before execution, obtain explicit repository-owner authorization, create and verify a mirror backup, freeze collaboration, notify every contributor, and rotate any credentials found by scanning. Then run `git filter-repo --invert-paths` once for every manifest path against the mirror, verify the rewritten object graph, force-push all affected refs, and require every collaborator and deployment to reclone. Retain the pre-rewrite mirror offline until the rewritten remote has been independently verified.

CI runs `scripts/scan_private_history.py`. It is expected to fail until the separately authorized rewrite is completed; deleting files only from the current checkout is not sufficient.
