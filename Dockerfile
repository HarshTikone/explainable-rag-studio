FROM python:3.11-slim
ARG PREFETCH_MODELS=false
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ARG GROUNDING_MODEL=cross-encoder/nli-deberta-v3-xsmall
ARG GROUNDING_MODEL_REVISION=a150876415327c80daeff35ca6f68f5ed8cf5c24
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8501 PYTHONPATH=/app RERANKER_MODEL=${RERANKER_MODEL} GROUNDING_MODEL=${GROUNDING_MODEL} GROUNDING_MODEL_REVISION=${GROUNDING_MODEL_REVISION}
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 postgresql-client tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .
RUN if [ "$PREFETCH_MODELS" = "true" ]; then python scripts/prefetch_models.py; fi
RUN addgroup --system rag && adduser --system --ingroup rag --home /app rag && chown -R rag:rag /app
USER rag
EXPOSE 8000 8501
CMD ["sh", "deploy/start-service.sh", "streamlit"]
