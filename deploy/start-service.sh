#!/bin/sh
set -eu

. "$(dirname "$0")/env-secrets.sh"

case "${1:-}" in
  migrate)
    alembic upgrade head
    ;;
  api)
    exec uvicorn api:app --host 0.0.0.0 --port 8000
    ;;
  streamlit)
    exec streamlit run app/Home.py --server.address=0.0.0.0 --server.port=8501
    ;;
  worker)
    exec rq worker ingestion --url "$REDIS_URL" --with-scheduler
    ;;
  *)
    echo "Unknown service: ${1:-}" >&2
    exit 64
    ;;
esac
