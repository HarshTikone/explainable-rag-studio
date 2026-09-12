#!/bin/sh
# Shared secret-to-env resolution for the app image. Sourced (not executed)
# by start-service.sh's own process and by one-off `docker compose exec`
# commands, since `docker exec` only sees the container's original
# Dockerfile/compose environment, not variables a running PID 1 exported
# for itself at boot.
set -eu

read_secret() {
  tr -d '\r\n' < "/run/secrets/$1"
}

export API_KEY_PEPPER="$(read_secret api_key_pepper)"
export AUDIT_HMAC_KEY="$(read_secret audit_hmac_key)"
export OBJECT_STORE_ACCESS_KEY="$(read_secret minio_access_key)"
export OBJECT_STORE_SECRET_KEY="$(read_secret minio_secret_key)"
export DATABASE_URL="postgresql+psycopg://rag_app:$(read_secret rag_app_password)@postgres:5432/rag"
export REDIS_URL="redis://:$(read_secret redis_password)@redis:6379/0"
export OBJECT_MASTER_KEY_FILE=/run/secrets/object_master_key
export OIDC_CLIENT_SECRET_FILE=/run/secrets/oidc_client_secret
