"""Lazy, injectable cross-encoder reranking."""
from __future__ import annotations

from threading import Lock
from typing import List, Protocol, Sequence


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, documents: Sequence[str]) -> List[float]: ...


class RerankerUnavailableError(RuntimeError):
    pass


class CrossEncoderReranker:
    """CPU-first wrapper that loads the model only when reranking is requested."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        max_length: int = 512,
        backend: str = "torch",
        model_revision: str = "",
        onnx_file: str = "",
        cpu_threads: int = 2,
    ):
        if backend not in {"torch", "onnx"}:
            raise ValueError("Reranker backend must be 'torch' or 'onnx'.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.backend = backend
        self.model_revision = model_revision
        self.onnx_file = onnx_file
        self.cpu_threads = max(1, cpu_threads)
        self._model = None
        self._lock = Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        if self.backend == "onnx":
                            from .onnx_cross_encoder import NativeOnnxCrossEncoder
                            self._model = NativeOnnxCrossEncoder(
                                self.model_name, self.model_revision, self.onnx_file,
                                self.max_length, self.cpu_threads,
                            )
                        else:
                            from sentence_transformers import CrossEncoder
                            import torch
                            torch.set_num_threads(self.cpu_threads)
                            try:
                                torch.set_num_interop_threads(1)
                            except RuntimeError:
                                pass
                            self._model = CrossEncoder(
                                self.model_name,
                                revision=self.model_revision or None,
                                max_length=self.max_length,
                                device="cpu",
                            )
                    except Exception as exc:
                        raise RerankerUnavailableError(f"Could not load reranker {self.model_name}: {exc}") from exc
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        try:
            scores = self._load().predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except RerankerUnavailableError:
            raise
        except Exception as exc:
            raise RerankerUnavailableError(f"Reranking failed: {exc}") from exc
        if getattr(self._model, "native_onnx", False):
            import numpy as np
            scores = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype="float64")))
        return [float(score) for score in scores]


_default_rerankers = {}
_default_lock = Lock()


def get_default_reranker(
    model_name: str,
    batch_size: int = 16,
    max_length: int = 512,
    backend: str | None = None,
    model_revision: str | None = None,
    onnx_file: str | None = None,
    cpu_threads: int | None = None,
) -> CrossEncoderReranker:
    from .config import SETTINGS

    resolved = (
        backend or SETTINGS.reranker_backend,
        model_revision if model_revision is not None else SETTINGS.reranker_model_revision,
        onnx_file if onnx_file is not None else SETTINGS.reranker_onnx_file,
        cpu_threads or SETTINGS.model_cpu_threads,
    )
    key = (model_name, batch_size, max_length, *resolved)
    with _default_lock:
        if key not in _default_rerankers:
            _default_rerankers[key] = CrossEncoderReranker(model_name, batch_size, max_length, *resolved)
        return _default_rerankers[key]
