"""Small native ONNX cross-encoder runtime with bounded tokenization reuse."""
from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Sequence

import numpy as np


class NativeOnnxCrossEncoder:
    """Run a pinned Hugging Face ONNX file without the Optimum compatibility layer."""

    native_onnx = True

    def __init__(
        self,
        model_name: str,
        revision: str,
        file_name: str,
        max_length: int,
        cpu_threads: int,
        cache_size: int = 128,
    ):
        if not revision or not file_name:
            raise ValueError("Native ONNX models require a pinned revision and file name.")
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        model_path = hf_hub_download(repo_id=model_name, filename=file_name, revision=revision)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, cpu_threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.max_length = max_length
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.cache_size = max(1, cache_size)
        self._token_cache: OrderedDict[tuple, dict[str, np.ndarray]] = OrderedDict()
        self._cache_lock = Lock()

    def _features(self, pairs: Sequence[tuple[str, str]]) -> dict[str, np.ndarray]:
        key = tuple(pairs)
        with self._cache_lock:
            cached = self._token_cache.get(key)
            if cached is not None:
                self._token_cache.move_to_end(key)
                return cached
        encoded = self.tokenizer(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        features = {
            name: np.asarray(value, dtype=np.int64)
            for name, value in encoded.items()
            if name in self.input_names
        }
        with self._cache_lock:
            self._token_cache[key] = features
            self._token_cache.move_to_end(key)
            while len(self._token_cache) > self.cache_size:
                self._token_cache.popitem(last=False)
        return features

    def predict(self, pairs, *, batch_size: int = 16, **_kwargs) -> np.ndarray:
        materialized = list(pairs)
        if not materialized:
            return np.asarray([], dtype=np.float32)
        features = self._features(materialized)
        outputs = []
        for start in range(0, len(materialized), max(1, batch_size)):
            stop = min(len(materialized), start + max(1, batch_size))
            batch = {name: value[start:stop] for name, value in features.items()}
            outputs.append(np.asarray(self.session.run(None, batch)[0]))
        result = np.concatenate(outputs, axis=0)
        return result.squeeze(-1) if result.ndim == 2 and result.shape[1] == 1 else result
