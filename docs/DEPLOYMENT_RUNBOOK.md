# Deployment runbook

`docs/PRODUCTION_PLATFORM.md` covers the architecture and the cutover procedure. This is the
operational companion: what to actually *do* — deploy, roll back, respond to each fail-closed
condition, rotate secrets, and run the recovery exercise Milestone 6's exit gate calls for.
Every command here is one already exercised for real in `release-quality.yml`, not new,
unverified procedure.

## Deploying

1. `python scripts/generate_dev_secrets.py` — or, for a real environment, generate the same nine
   secret files (`api_key_pepper`, `audit_hmac_key`, `rag_app_password`, `postgres_password`,
   `minio_access_key`, `minio_secret_key`, `object_master_key`, `oidc_client_secret`,
   `redis_password`, `keycloak_db_password`, `keycloak_admin_password`) through whatever secret
   manager the target platform provides, mounted at `/run/secrets/<name>` exactly as
   `docker-compose.yml`'s `secrets:` block expects. Never bake these into the image or commit them.
2. `docker compose build && docker compose up -d`. The `migrate` service runs `alembic upgrade`
   and must complete (`service_completed_successfully`) before `api` starts — this is enforced by
   `depends_on`, not just convention.
3. Wait for `/ready` (api) and `/_stcore/health` (streamlit); both fail closed (see below) if any
   dependency isn't reachable, so a 200 here is a real readiness signal, not just "the process
   started."
4. Bootstrap the first owner: `docker compose exec api python scripts/bootstrap_security.py
   --organization "..." --email owner@...`. The returned API key is shown once — store it in
   whatever secret manager the deploying team uses, not in a ticket or chat log.

## Cutover from the legacy (local FAISS/SQLite) profile

Follow `docs/PRODUCTION_PLATFORM.md`'s Cutover section exactly: `scripts/migrate_platform.py
plan` → review → `execute` → `validate`, then flip `PLATFORM_MODE=postgres`. Keep the old
SQLite/FAISS data read-only for one release — it's the rollback path if the new profile needs to
be backed out.

## Rollback

Because there's no dual-write period, rollback means reverting `PLATFORM_MODE` to `legacy` and
pointing back at the retained read-only SQLite/FAISS data from before cutover — not a database
downgrade migration. If the issue surfaces *after* new writes have landed in Postgres, those
writes are not automatically replayed back to the legacy store; treat rollback as a
last-resort measure taken before meaningful new production traffic, not a routine undo.

## The recovery exercise (verified, not hypothetical)

Milestone 6's exit gate calls for "a documented recovery exercise." This one isn't a paper
procedure — it's the exact sequence `release-quality.yml`'s "Backup and restore evidence" step
runs against the real compose stack on every CI run:

1. `scripts/platform_backup.py` (run as `rag_owner`, the only role permitted to bypass row-level
   security for a complete `pg_dump`) produces a custom-format Postgres dump, uploads it through
   the encrypted object-store adapter (client-side AES-256-GCM envelope encryption — the object
   store never sees plaintext), and returns an `object_key`.
2. `scripts/restore_validate.py <object_key> --restore-url <isolated-database-url>` downloads and
   decrypts that dump, restores it into a separate, freshly-created database (never over the
   source), and validates the restored data against the four-hour RTO target.
3. This has passed on every completed `release-quality.yml` run since the pipeline first got
   working infrastructure (2026-09-12's `20260912T154030Z` retained release onward) — it is a
   real, repeatedly-exercised procedure, not a documented-but-never-run one.

To run it manually against a real deployment: same two commands, pointed at the real
`ADMIN_DATABASE_URL` and a scratch restore database, per `deploy/env-secrets.sh`'s resolution
logic. Retention: daily backups (documented 24-hour RPO), 90-day release-evidence retention.
Restore testing must run at least once per release — the CI step above satisfies that on every
push to `main`, so a manual pre-release run is a re-confirmation, not the only line of defense.

## Responding to each fail-closed condition

`docs/PRODUCTION_PLATFORM.md` lists what fails closed; here's what an operator does about each:

- **`/ready` fails (missing database/Redis/OIDC/object-store/secret configuration)** — check
  which dependency's health check is failing (`docker compose ps`, then the specific service's
  logs). This is almost always a secret-mounting or dependency-startup-order problem, not
  application logic — `api` won't even attempt to serve until every declared dependency is
  healthy.
- **Redis loss → `503` on protected operations** — expected, not a bug: rate-limit enforcement
  can't be proven without Redis, and the limiter fails closed rather than silently allowing
  unbounded traffic (`backend/rate_limit.py`). Restore Redis; no data loss risk, since job state
  lives in Postgres and Redis is explicitly non-authoritative.
- **Object checksum/authentication failure** — treat as a real integrity incident, not a retry
  candidate: it means either the encrypted object was corrupted in transit/at rest, or someone
  attempted to tamper with ciphertext they don't hold the key for. Don't retry-and-ignore; find
  out which object and why before assuming it's transient.
- **A cross-tenant database/retrieval result** — this should be structurally impossible (two
  independent enforcement layers, per `docs/THREAT_MODEL.md`'s threat #1), so if it's ever
  observed, treat it as a critical security incident requiring immediate investigation, not a bug
  ticket — start with `verify_audit_chain()` to establish what actually happened.

## Secret rotation

Rotate a secret by regenerating its file and restarting only the services that mount it (check
`docker-compose.yml`'s per-service `secrets:` list — most secrets go to every app service, but
`postgres_password` is deliberately scoped to `api` alone). API keys don't need coordinated
rotation the way shared secrets do: `revoke_api_key` plus issuing a new one
(`create_key_for_context`) is immediate and per-key, with no restart required.

## Public demo deployment (Render)

Everything above is the multi-tenant `PLATFORM_MODE=postgres` reference stack
(`docker-compose.yml`). Milestone 7's public sanitized demo target is deliberately different and
much simpler: the `legacy` profile — one container, local FAISS/SQLite, no Postgres/Redis/
MinIO/Keycloak — served via `render.yaml` at the repo root.

**Why legacy mode for the public demo, not the full production stack**: the production profile
needs OIDC login (Keycloak) before anyone can do anything, which is the wrong first impression
for a portfolio demo meant to be tried in one click. Legacy mode with `SECURITY_MODE=demo` gives
anonymous visitors a `viewer`-role context (`backend/security_models.py`'s `ROLE_SCOPES`) —
`query` and `documents:read` only, no upload/write access — automatically, with no login, and
query volume is already rate-limited for anonymous callers
(`backend/rate_limit.py::MemoryDemoRateLimiter`, 60 requests/minute shared across all anonymous
visitors by default). Nothing new had to be built for this; the safety rails already existed for
exactly this use case.

### Deploy steps

1. In the Render dashboard, create a new Blueprint from this repository (Render auto-detects
   `render.yaml` at the repo root). Since GitHub is already connected to the Render account, this
   is a few clicks — no manual service configuration needed.
2. Render builds the existing `Dockerfile` with `PREFETCH_MODELS=true` (bakes the reranker and
   grounding models into the image at build time, so the first visitor after a deploy doesn't hit
   a slow cold-load) and starts it on the `free` plan. **The free plan is a cost-free deployment
   target, not a claim that the full local-model stack fits comfortably within its memory limit**
   — watch the first deploy's memory usage and treat an out-of-memory result as a sizing signal.
3. Add `GEMINI_API_KEY` as a secret environment variable in the Render dashboard after the first
   deploy (`render.yaml` deliberately leaves it as `sync: false` — never commit a real key).
   Optional: without it, claim generation falls back to the deterministic extractive mode
   (`docs/CLAIM_LEVEL_GROUNDING.md`) and the demo still fully works.
4. The free Blueprint deliberately has no persistent disk. The Docker build therefore runs
   `scripts/build_demo_index.py` and bakes the bundled public demo index into the image alongside
   the model cache. Every fresh instance starts query-ready, while anonymous `viewer` access
   remains unable to ingest or modify documents.
5. Link the live URL from `README.md`'s demo section, and from
   `docs/DEMO_WALKTHROUGH_SCRIPT.md` once that's recorded against the live instance.

### What isn't verified

This session has no Render account access, so none of the above has been confirmed against a
real Render deploy — it's reasoned from `render.yaml`'s documented schema, this repo's existing
`Dockerfile`/`deploy/start-service.sh`, and the legacy-mode code paths already read in
`app/security_ui.py` and `api.py`. Watch the real build/deploy logs in the Render dashboard; if
something fails, the failure and its logs are the next real input to fix from — same
build-fix-reverify discipline as everything else in this document, just running on Render's
infrastructure instead of GitHub Actions.
