FROM python:3.11-slim
ARG PREFETCH_MODELS=false
ARG BUILD_DEMO_INDEX=false
ARG LOW_MEMORY_DEMO=false
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG GROUNDING_MODEL=cross-encoder/nli-deberta-v3-xsmall
ARG GROUNDING_MODEL_REVISION=a150876415327c80daeff35ca6f68f5ed8cf5c24
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8501 PYTHONPATH=/app LOW_MEMORY_DEMO=${LOW_MEMORY_DEMO} RERANKER_MODEL=${RERANKER_MODEL} GROUNDING_MODEL=${GROUNDING_MODEL} GROUNDING_MODEL_REVISION=${GROUNDING_MODEL_REVISION}
WORKDIR /app
# The public image needs only certificates and FAISS's OpenMP runtime. The full
# profile additionally pins the PostgreSQL 16 client so backup/restore remains
# compatible with the pg16 server in docker-compose.yml.
RUN if [ "$LOW_MEMORY_DEMO" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg libgomp1 tesseract-ocr && \
        install -d /usr/share/postgresql-common/pgdg && \
        curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc && \
        . /etc/os-release && \
        echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt ${VERSION_CODENAME}-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
        apt-get update && apt-get install -y --no-install-recommends postgresql-client-16 && \
        rm -rf /var/lib/apt/lists/*; \
    fi
COPY requirements.txt requirements-public.txt ./
RUN pip install --upgrade pip && \
    if [ "$LOW_MEMORY_DEMO" = "true" ]; then \
        pip install -r requirements-public.txt; \
    else \
        pip install -r requirements.txt; \
    fi
COPY . .
RUN if [ "$PREFETCH_MODELS" = "true" ]; then \
        python scripts/prefetch_models.py; \
    fi
RUN if [ "$BUILD_DEMO_INDEX" = "true" ]; then \
        python scripts/build_demo_index.py; \
    fi
RUN addgroup --system rag && adduser --system --ingroup rag --home /app rag && chown -R rag:rag /app
USER rag
EXPOSE 8000 8501
CMD ["sh", "deploy/start-service.sh", "streamlit"]
