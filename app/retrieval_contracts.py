"""Typed contracts for bounded patient-scoped hybrid retrieval.

Authority-bearing identifiers are constructed by the server, never by an LLM.
The hybrid workflow is evaluation-only in this release and cannot enable itself
for live patient retrieval.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HYBRID_RETRIEVAL_REVISION = "hybrid-retrieval-eval-v3"


class RetrievalMode(str, Enum):
    FTS5 = "fts5"
    HYBRID_EVAL = "hybrid_eval"

class RerankStrategy(str, Enum):
    NONE = "none"
    REPLACE = "replace"
    RANK_BLEND = "rank_blend"
    SCORE_BLEND = "score_blend"
    ANCHORED_SCORE_BLEND = "anchored_score_blend"
    PREFERRED_KIND_RESCUE = "preferred_kind_rescue"



class ClinicalAccessScope(BaseModel):
    """Server-constructed authorization scope; never model-generated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    patient_key: str = Field(min_length=1, max_length=128)
    allowed_document_ids: frozenset[str] = Field(min_length=1)
    chart_revision: int | None = Field(default=None, ge=1)
    thread_id: str | None = Field(default=None, max_length=128)

    @field_validator("allowed_document_ids")
    @classmethod
    def no_empty_ids(cls, value: frozenset[str]) -> frozenset[str]:
        if any(not str(item).strip() for item in value):
            raise ValueError("allowed_document_ids cannot contain empty identifiers")
        return value


class HybridRetrievalPolicy(BaseModel):
    """Fail-closed retrieval policy for local evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: RetrievalMode = RetrievalMode.FTS5
    lexical_k: int = Field(default=40, ge=1, le=100)
    dense_k: int = Field(default=40, ge=1, le=100)
    fusion_k: int = Field(default=20, ge=1, le=100)
    rerank_k: int = Field(default=10, ge=1, le=50)
    final_k: int = Field(default=5, ge=1, le=20)
    rrf_constant: int = Field(default=60, ge=1, le=1000)
    use_reranker: bool = True
    rerank_strategy: RerankStrategy = RerankStrategy.RANK_BLEND
    lexical_rank_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    dense_rank_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    fusion_rank_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    rerank_rank_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    lexical_anchor_depth: int = Field(default=3, ge=0, le=10)
    lexical_anchor_bonus: float = Field(default=0.08, ge=0.0, le=0.5)
    lexical_score_weight: float = Field(default=0.50, ge=0.0, le=1.0)
    dense_score_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    rerank_score_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    preferred_kind_gate_depth: int = Field(default=5, ge=1, le=20)
    remote_embeddings_allowed: Literal[False] = False
    remote_reranking_allowed: Literal[False] = False
    remote_vector_db_allowed: Literal[False] = False
    external_tracing_allowed: Literal[False] = False
    persist_vectors: Literal[False] = False

    @model_validator(mode="after")
    def validate_candidate_sizes(self):
        if self.final_k > self.rerank_k:
            raise ValueError("final_k cannot exceed rerank_k")
        if self.rerank_k > self.fusion_k:
            raise ValueError("rerank_k cannot exceed fusion_k")
        if self.fusion_k > max(self.lexical_k, self.dense_k):
            raise ValueError("fusion_k cannot exceed both first-stage candidate counts")
        if self.use_reranker and self.rerank_strategy == RerankStrategy.NONE:
            raise ValueError("use_reranker=True is incompatible with rerank_strategy='none'")
        reranker_required = {
            RerankStrategy.REPLACE,
            RerankStrategy.RANK_BLEND,
            RerankStrategy.SCORE_BLEND,
            RerankStrategy.ANCHORED_SCORE_BLEND,
        }
        if not self.use_reranker and self.rerank_strategy in reranker_required:
            raise ValueError(f"rerank_strategy={self.rerank_strategy.value!r} requires use_reranker=True")
        rank_weights = (
            self.lexical_rank_weight + self.dense_rank_weight +
            self.fusion_rank_weight + self.rerank_rank_weight
        )
        if self.rerank_strategy == RerankStrategy.RANK_BLEND and abs(rank_weights - 1.0) > 1e-6:
            raise ValueError("rank-blend weights must sum to 1.0")
        score_weights = self.lexical_score_weight + self.dense_score_weight + self.rerank_score_weight
        if self.rerank_strategy in {
            RerankStrategy.SCORE_BLEND,
            RerankStrategy.ANCHORED_SCORE_BLEND,
            RerankStrategy.PREFERRED_KIND_RESCUE,
        } and self.use_reranker and abs(score_weights - 1.0) > 1e-6:
            raise ValueError("score-blend weights must sum to 1.0 when the reranker is enabled")
        if self.rerank_strategy == RerankStrategy.PREFERRED_KIND_RESCUE and not self.use_reranker:
            local_weights = self.lexical_score_weight + self.dense_score_weight
            if local_weights <= 0.0:
                raise ValueError("preferred-kind rescue without a reranker needs lexical or dense score weight")
        return self


class HybridRetrievalRequest(BaseModel):
    """Request content. Patient authority lives only in `scope`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=2, max_length=2000)
    scope: ClinicalAccessScope
    policy: HybridRetrievalPolicy = Field(default_factory=HybridRetrievalPolicy)


class CandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    source_kind: str
    lexical_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    fusion_score: float | None = None
    rerank_score: float | None = None
    lexical_score: float | None = None
    dense_score: float | None = None
    blended_score: float | None = None
    final_rank: int | None = Field(default=None, ge=1)


class HybridRetrievalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: str = HYBRID_RETRIEVAL_REVISION
    query_id: str
    mode: RetrievalMode
    backend: str
    candidates: list[CandidateScore]
    authorized_document_count: int = Field(ge=0)
    rejected_document_ids: list[str] = Field(default_factory=list)
    patient_leakage_count: int = Field(default=0, ge=0)
    semantic_available: bool = False
    reranker_available: bool = False
    semantic_used: bool = False
    model_downloads_allowed: Literal[False] = False
    remote_calls_allowed: Literal[False] = False
    external_tracing_allowed: Literal[False] = False
