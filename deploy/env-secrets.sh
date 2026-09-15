#!/bin/sh
# Shared secret-to-env resolution for the app image. Sourced (not executed)
# by start-service.sh's own process and by one-off `docker compose exec`
# commands, since `docker exec` only sees the container's original
# Dockerfile/compose environment, not variables a running PID 1 exported
# for itself at boot.
#
# Only the "postgres" platform mode (docker-compose.yml's reference stack)
# uses Docker secret files under /run/secrets/ -- the default "legacy" mode
# (backend/config.py: PLATFORM_MODE defaults to "legacy") runs on local
# FAISS/SQLite and needs none of this. Skip entirely outside postgres mode
# so a plain `docker run` or a host platform that injects its own env vars
# (Render, Fly.io, Railway, ...) doesn't crash on startup trying to read
# secret files that were never mounted.
set -eu

if [ "${PLATFORM_MODE:-legacy}" = "postgres" ]; then
  read_secret() {
    tr -d '\r\n' < "/run/secrets/$1"
  }

  export API_KEY_PEPPER="$(read_secret api_key_pepper)"
  export AUDIT_HMAC_KEY="$(read_secret audit_hmac_key)"
  export OBJECT_STORE_ACCESS_KEY="$(read_secret minio_access_key)"
  export OBJECT_STORE_SECRET_KEY="$(read_secret minio_secret_key)"
  export DATABASE_URL="postgresql+psycopg://rag_app:$(read_secret rag_app_password)@postgres:5432/rag"
  export REDIS_URL="redis://:$(read_secret redis_password)@redis:6379/0"
  # rag_app is deliberately NOBYPASSRLS (backend/database.py enforces tenant
  # isolation via FORCE ROW LEVEL SECURITY), so pg_dump/pg_restore need the
  # postgres superuser instead -- only present when /run/secrets/postgres_password
  # is mounted (currently just the api container, for backup/restore scripts).
  if [ -r /run/secrets/postgres_password ]; then
    export ADMIN_DATABASE_URL="postgresql+psycopg://rag_owner:$(read_secret postgres_password)@postgres:5432/rag"
  fi
  export OBJECT_MASTER_KEY_FILE=/run/secrets/object_master_key
  export OIDC_CLIENT_SECRET_FILE=/run/secrets/oidc_client_secret
fi
