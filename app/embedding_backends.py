"""Strict-local embedding backends for hybrid retrieval evaluation.

FastEmbed can download models by default. This adapter always sets
`local_files_only=True`; a model must already exist in the configured cache or
be supplied with `specific_model_path`. No clinical text is sent over HTTP.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    name: str
    available: bool
    error: str | None

    def embed_query(self, text: str) -> np.ndarray: ...
    def embed_documents(self, texts: list[str]) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class LocalEmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    cache_dir: str | None = None
    specific_model_path: str | None = None
    threads: int | None = None

    def validate_paths(self) -> None:
        if self.specific_model_path and not Path(self.specific_model_path).is_dir():
            raise ValueError(f"Embedding model path does not exist: {self.specific_model_path}")
        if self.cache_dir and not Path(self.cache_dir).is_dir():
            raise ValueError(f"Embedding cache directory does not exist: {self.cache_dir}")


class FastEmbedBackend:
    name = "fastembed_dense"

    def __init__(self, config: LocalEmbeddingConfig | None = None):
        self.config = config or LocalEmbeddingConfig(
            model_name=os.getenv("RETRIEVAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            cache_dir=os.getenv("RETRIEVAL_EMBEDDING_CACHE") or None,
            specific_model_path=os.getenv("RETRIEVAL_EMBEDDING_MODEL_PATH") or None,
        )
        self._model = None
        self.error: str | None = None
        try:
            self.config.validate_paths()
            from fastembed import TextEmbedding
            kwargs = {
                "model_name": self.config.model_name,
                "local_files_only": True,
            }
            if self.config.cache_dir:
                kwargs["cache_dir"] = self.config.cache_dir
            if self.config.specific_model_path:
                kwargs["specific_model_path"] = self.config.specific_model_path
            if self.config.threads:
                kwargs["threads"] = self.config.threads
            self._model = TextEmbedding(**kwargs)
        except Exception as exc:  # optional evaluation dependency/model
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._model is not None

    @staticmethod
    def _unit(vector: Iterable[float]) -> np.ndarray:
        array = np.asarray(list(vector), dtype=np.float32)
        norm = float(np.linalg.norm(array))
        return array if norm == 0.0 else array / norm

    def embed_query(self, text: str) -> np.ndarray:
        if not self._model:
            raise RuntimeError(self.error or "FastEmbed embedding model is unavailable")
        return self._unit(next(iter(self._model.query_embed(text))))

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        if not self._model:
            raise RuntimeError(self.error or "FastEmbed embedding model is unavailable")
        return [self._unit(vector) for vector in self._model.embed(texts)]


class DeterministicHashEmbeddingBackend:
    """Test-only local embedding backend with no external dependency.

    It is intentionally not exposed as a production candidate. It allows the
    graph/isolation behavior to be exercised when FastEmbed is unavailable in
    the build environment.
    """
    name = "deterministic_hash_test_embedding"
    available = True
    error = None

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions
        self.seen_texts: list[str] = []

    def _embed(self, text: str) -> np.ndarray:
        import hashlib
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in text.casefold().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0 else vector / norm

    def embed_query(self, text: str) -> np.ndarray:
        self.seen_texts.append(text)
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        self.seen_texts.extend(texts)
        return [self._embed(text) for text in texts]


def cosine_scores(query_vector: np.ndarray, document_vectors: list[np.ndarray]) -> list[float]:
    return [float(np.dot(query_vector, vector)) for vector in document_vectors]


def make_langchain_embeddings(backend: EmbeddingBackend):
    """Return a LangChain-Core `Embeddings` adapter when that package exists."""
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError as exc:  # optional dependency
        raise RuntimeError("langchain-core is not installed; install requirements-ml.txt") from exc

    class LocalAdapter(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [vector.astype(float).tolist() for vector in backend.embed_documents(texts)]

        def embed_query(self, text: str) -> list[float]:
            return backend.embed_query(text).astype(float).tolist()

    return LocalAdapter()
