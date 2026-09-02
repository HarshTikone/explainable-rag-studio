# Northstar Operations Guide

The production service exposes a health endpoint at /health. Telemetry separates retrieval latency from generation latency and reports p50 and p95. Experiment artifacts are stored under outputs/experiments using a unique experiment identifier. The container listens on the platform-provided PORT and defaults to 8501 locally. If Gemini is unavailable, the application uses a deterministic extractive fallback so retrieval benchmarks can continue without an API key.
