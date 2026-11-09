"""Strict-local reranking backends for evaluation-only hybrid retrieval."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol


class Reranker(Protocol):
    name: str
    available: bool
    error: str | None
    def score(self, query: str, documents: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class LocalRerankerConfig:
    model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    cache_dir: str | None = None
    specific_model_path: str | None = None
    threads: int | None = None

    def validate_paths(self) -> None:
        if self.specific_model_path and not Path(self.specific_model_path).is_dir():
            raise ValueError(f"Reranker model path does not exist: {self.specific_model_path}")
        if self.cache_dir and not Path(self.cache_dir).is_dir():
            raise ValueError(f"Reranker cache directory does not exist: {self.cache_dir}")


class FastEmbedReranker:
    name = "fastembed_cross_encoder"

    def __init__(self, config: LocalRerankerConfig | None = None):
        self.config = config or LocalRerankerConfig(
            model_name=os.getenv("RETRIEVAL_RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
            cache_dir=os.getenv("RETRIEVAL_RERANKER_CACHE") or None,
            specific_model_path=os.getenv("RETRIEVAL_RERANKER_MODEL_PATH") or None,
        )
        self._model = None
        self.error: str | None = None
        try:
            self.config.validate_paths()
            from fastembed.rerank.cross_encoder import TextCrossEncoder
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
            self._model = TextCrossEncoder(**kwargs)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self._model is not None

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not self._model:
            raise RuntimeError(self.error or "FastEmbed reranker is unavailable")
        return [float(score) for score in self._model.rerank(query, documents)]


class KeywordOverlapTestReranker:
    """Dependency-free test reranker; not a production candidate."""
    name = "keyword_overlap_test_reranker"
    available = True
    error = None

    def __init__(self):
        self.seen_documents: list[str] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        import re
        q = set(re.findall(r"[a-z0-9]+", query.casefold()))
        self.seen_documents.extend(documents)
        result = []
        for document in documents:
            d = set(re.findall(r"[a-z0-9]+", document.casefold()))
            result.append(len(q & d) / max(1, len(q | d)))
        return result
