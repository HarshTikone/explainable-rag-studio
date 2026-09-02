# Portable production platform

The production profile replaces process-local persistence with PostgreSQL/pgvector, Redis/RQ, private S3-compatible storage, and generic OIDC. The original SQLite/FAISS profile remains available as `PLATFORM_MODE=legacy` for the public offline demo and as a one-release read-only rollback source.

## Architecture

- FastAPI is the only trusted application boundary. Streamlit uses `API_BASE_URL` and does not open databases, indexes, queues, or object storage in production mode.
- PostgreSQL is authoritative for organizations, memberships, credentials, lifecycle state, chunks, vectors, reviews, telemetry, quotas, retention, and audit records.
- Every tenant table has forced row-level security. FastAPI sets `app.organization_id` and `app.actor_user_id` transaction-locally from the authenticated security context.
- pgvector HNSW performs dense retrieval. BM25 is rebuilt from the tenant-filtered PostgreSQL chunk set and can be invalidated between API replicas through Redis.
- Redis transports idempotent ingestion jobs and enforces atomic token buckets. Job state remains in PostgreSQL so Redis can be replaced without data recovery.
- MinIO provides the local S3 contract. Private objects use randomized tenant-prefixed keys, S3 server-side encryption, and application-level AES-256-GCM envelope encryption.
- Keycloak is the local OIDC provider. Production accepts any conforming issuer with pinned issuer and audience validation. API keys remain available for automation. SCIM is deferred.

## Local reference deployment

1. Install Docker with Compose support.
2. Run `python scripts/generate_dev_secrets.py`. The generated `.secrets/` directory is ignored by Git.
3. Run `docker compose build` and `docker compose up -d`.
4. Wait for `http://localhost:8000/ready` and `http://localhost:8501/_stcore/health`.
5. Bootstrap the first owner inside the API container:

   `docker compose exec api python scripts/bootstrap_security.py --organization "Example" --email owner@example.invalid`

The bootstrap secret is printed once. Bind an OIDC subject to a membership through `POST /members/{user_id}/oidc`. The organization is resolved from the credential and verified against membership; it is never accepted from ordinary request parameters.

## Cutover

Run `scripts/migrate_platform.py plan` first. The plan rejects mixed-tenant indexes and records exact chunk counts, dimensions, and fingerprints. After review, run `execute`, then `validate`. Lifecycle mutations are paused for the final delta. There is no dual-write period. Set `PLATFORM_MODE=postgres` only after validation, and retain the old SQLite/FAISS data read-only for one release.

## Recovery and retention

`scripts/platform_backup.py` sends a custom PostgreSQL dump through the encrypted object-store adapter. `scripts/restore_validate.py` refuses to restore over the source database and checks the isolated database against the four-hour RTO target. Redis is non-authoritative.

Default retention is 365 days for audit events, 90 days for release evidence, and 30 days for deleted or quarantined content. Restore testing must run at least once per release; daily backups provide the documented 24-hour RPO.

## Fail-closed behavior

- Missing database, Redis, OIDC, object-store, or mounted-secret configuration makes `/ready` fail.
- Redis loss returns `503` for protected operations because rate-limit enforcement cannot be proven.
- Object authentication or checksum failure prevents parsing and indexing.
- A cross-tenant database or retrieval result aborts the request before reranking, generation, citations, review persistence, or serialization.
- `hybrid_rrf` and strict supported-only grounding remain the production defaults until retained ML artifacts pass every promotion gate.
