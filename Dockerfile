FROM python:3.11-slim
ARG PREFETCH_MODELS=false
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG GROUNDING_MODEL=cross-encoder/nli-deberta-v3-xsmall
ARG GROUNDING_MODEL_REVISION=a150876415327c80daeff35ca6f68f5ed8cf5c24
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8501 PYTHONPATH=/app RERANKER_MODEL=${RERANKER_MODEL} GROUNDING_MODEL=${GROUNDING_MODEL} GROUNDING_MODEL_REVISION=${GROUNDING_MODEL_REVISION}
WORKDIR /app
# Debian's generic "postgresql-client" metapackage tracks whatever major
# version ships with the base image, which drifts ahead of the pg16 server
# (pgvector/pgvector:pg16 in docker-compose.yml) as Debian releases roll
# forward -- a newer pg_restore then emits session setup the pg16 server
# rejects ("unrecognized configuration parameter"). Pin postgresql-client-16
# via the PGDG apt repo so client and server major versions always match.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg libgomp1 tesseract-ocr && \
    install -d /usr/share/postgresql-common/pgdg && \
    curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc && \
    . /etc/os-release && \
    echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt ${VERSION_CODENAME}-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && apt-get install -y --no-install-recommends postgresql-client-16 && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
RUN if [ "$PREFETCH_MODELS" = "true" ]; then \
        python scripts/prefetch_models.py && \
        python scripts/build_demo_index.py; \
    fi
RUN addgroup --system rag && adduser --system --ingroup rag --home /app rag && chown -R rag:rag /app
USER rag
EXPOSE 8000 8501
CMD ["sh", "deploy/start-service.sh", "streamlit"]
