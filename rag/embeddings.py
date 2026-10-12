"""Local sentence embeddings with fastembed (ONNX, CPU-only, no API key).

The model (~130 MB) is downloaded from Hugging Face the first time it's used and cached
after that. If it can't be loaded (e.g. no internet on first run), the app falls back to
keyword search only, instead of crashing.
"""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingUnavailable(RuntimeError):
    pass


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception as e:  # network errors, missing package, etc.
                raise EmbeddingUnavailable(f"Could not load embedding model '{self.model_name}': {e}") from e
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._load().embed(texts)), dtype=np.float32)
        return _normalise(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        # bge models work best with this instruction prefix on the query side
        vec = np.array(list(self._load().query_embed(text)), dtype=np.float32)
        return _normalise(vec)[0]


def _normalise(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)
