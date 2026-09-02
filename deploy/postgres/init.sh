#!/bin/sh
set -eu

rag_password="$(tr -d '\r\n' < /run/secrets/rag_app_password)"
keycloak_password="$(tr -d '\r\n' < /run/secrets/keycloak_db_password)"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --set=rag_password="$rag_password" --set=keycloak_password="$keycloak_password" <<-'SQL'
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE ROLE rag_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS PASSWORD :'rag_password';
    GRANT CONNECT ON DATABASE rag TO rag_app;
    GRANT USAGE, CREATE ON SCHEMA public TO rag_app;
    CREATE ROLE keycloak LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD :'keycloak_password';
    CREATE DATABASE keycloak OWNER keycloak;
SQL
