"""Download release-time local models into the image cache."""
import os

from backend.config import SETTINGS
from backend.grounding import CrossEncoderNliVerifier
from backend.reranker import CrossEncoderReranker


model_name = os.getenv("RERANKER_MODEL", SETTINGS.reranker_model)
reranker = CrossEncoderReranker(
    model_name, max_length=512, backend=SETTINGS.reranker_backend,
    model_revision=SETTINGS.reranker_model_revision,
    onnx_file=SETTINGS.reranker_onnx_file, cpu_threads=SETTINGS.model_cpu_threads,
)
reranker._load()
print(f"Prefetched reranker: {model_name}@{SETTINGS.reranker_model_revision} ({SETTINGS.reranker_backend})")

grounding_model = os.getenv("GROUNDING_MODEL", SETTINGS.grounding_model)
grounding_revision = os.getenv("GROUNDING_MODEL_REVISION", SETTINGS.grounding_model_revision)
verifier = CrossEncoderNliVerifier(
    grounding_model, grounding_revision, max_length=512, backend=SETTINGS.grounding_backend,
    onnx_file=SETTINGS.grounding_onnx_file, cpu_threads=SETTINGS.model_cpu_threads,
)
verifier._load()
print(f"Prefetched grounding verifier: {grounding_model}@{grounding_revision} ({SETTINGS.grounding_backend})")
