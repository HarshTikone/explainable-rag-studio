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
