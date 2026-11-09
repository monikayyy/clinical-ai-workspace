"""Bounded patient-scoped hybrid retrieval workflow.

LangGraph is optional and evaluation-only in this release. The same node logic is
available through `run_reference()` so authorization, leakage, and scoring can
be tested even in a build environment where optional packages are unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
import os
import time
from typing import Any, TypedDict

from .embedding_backends import EmbeddingBackend, cosine_scores
from .rerankers import Reranker
from .retrieval_contracts import (
    CandidateScore,
    HybridRetrievalPolicy,
    HybridRetrievalRequest,
    HybridRetrievalResponse,
    RetrievalMode,
    RerankStrategy,
)
from .retrieval_eval import FTS5Retriever, RetrievalDocument, RetrievalHit, query_profile, reciprocal_rank_fusion


class RetrievalSecurityError(RuntimeError):
    pass


class OptionalDependencyUnavailable(RuntimeError):
    pass


class HybridGraphState(TypedDict, total=False):
    request: HybridRetrievalRequest
    documents: list[RetrievalDocument]
    scoped_documents: list[RetrievalDocument]
    lexical_hits: list[RetrievalHit]
    dense_hits: list[RetrievalHit]
    fused_hits: list[RetrievalHit]
    reranked_hits: list[RetrievalHit]
    final_hits: list[RetrievalHit]
    blended_scores: dict[str, float]
    rejected_document_ids: list[str]
    timings_ms: dict[str, float]
    semantic_used: bool


def external_tracing_enabled() -> bool:
    truthy = {"1", "true", "yes", "on"}
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        if os.getenv(name, "").strip().casefold() in truthy:
            return True
    return False


@dataclass
class PatientScopedHybridWorkflow:
    embedding_backend: EmbeddingBackend | None = None
    reranker: Reranker | None = None
    max_cached_document_vectors: int = 5000
    _document_vector_cache: dict[tuple[str, str, str], Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.fts = FTS5Retriever()

    @staticmethod
    def _document_fingerprint(document: RetrievalDocument) -> str:
        payload = (document.title + "\n" + document.text).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _rank_component(rank: int | None) -> float:
        return 0.0 if not rank else 1.0 / float(rank)

    @staticmethod
    def _minmax(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        low = min(values.values())
        high = max(values.values())
        if abs(high - low) < 1e-12:
            return {key: 1.0 for key in values}
        scale = high - low
        return {key: (value - low) / scale for key, value in values.items()}

    @staticmethod
    def _preferred_kind_gate_satisfied(request: HybridRetrievalRequest, lexical_hits: list[RetrievalHit]) -> bool:
        if request.policy.rerank_strategy != RerankStrategy.PREFERRED_KIND_RESCUE:
            return False
        preferred = set(query_profile(request.question).get("preferred_source_kinds") or ())
        if not preferred:
            return False
        depth = min(request.policy.preferred_kind_gate_depth, len(lexical_hits))
        return any(hit.source_kind in preferred for hit in lexical_hits[:depth])

    @staticmethod
    def _timed(state: HybridGraphState, name: str, started: float) -> None:
        timings = dict(state.get("timings_ms") or {})
        timings[name] = round((time.perf_counter() - started) * 1000.0, 3)
        state["timings_ms"] = timings

    def scope_node(self, state: HybridGraphState) -> dict[str, Any]:
        started = time.perf_counter()
        request = state["request"]
        scope = request.scope
        allowed = set(scope.allowed_document_ids)
        documents = list(state.get("documents") or [])
        rejected: list[str] = []
        scoped: list[RetrievalDocument] = []
        for document in documents:
            if document.patient_key != scope.patient_key or document.id not in allowed:
                rejected.append(document.id)
                continue
            scoped.append(document)
        # Fail closed rather than silently allowing a caller to mix another
        # patient's content into embedding/reranking inputs.
        if rejected:
            raise RetrievalSecurityError(
                "Retrieval input contained documents outside the authorized patient/source scope: "
                + ", ".join(sorted(rejected)[:8])
            )
        if not scoped:
            raise RetrievalSecurityError("Authorized retrieval scope contained no eligible documents.")
        result: HybridGraphState = {"scoped_documents": scoped, "rejected_document_ids": []}
        self._timed(result, "scope", started)
        return result

    def lexical_node(self, state: HybridGraphState) -> dict[str, Any]:
        started = time.perf_counter()
        request = state["request"]
        hits = self.fts.rank(
            request.question,
            state["scoped_documents"],
            limit=request.policy.lexical_k,
        )
        result: HybridGraphState = {"lexical_hits": hits}
        self._timed(result, "lexical", started)
        return result

    def dense_node(self, state: HybridGraphState) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.embedding_backend or not self.embedding_backend.available:
            result: HybridGraphState = {"dense_hits": [], "semantic_used": False}
            self._timed(result, "dense", started)
            return result
        request = state["request"]
        documents = state["scoped_documents"]
        query_vector = self.embedding_backend.embed_query(request.question)
        vectors = []
        misses: list[tuple[tuple[str, str, str], str]] = []
        for document in documents:
            key = (
                request.scope.patient_key,
                document.id,
                self._document_fingerprint(document),
            )
            vector = self._document_vector_cache.get(key)
            if vector is None:
                misses.append((key, document.title + "\n" + document.text))
                vectors.append(None)
            else:
                vectors.append(vector)
        if misses:
            new_vectors = self.embedding_backend.embed_documents([text for _, text in misses])
            by_key = {key: vector for (key, _), vector in zip(misses, new_vectors)}
            for key, vector in by_key.items():
                self._document_vector_cache[key] = vector
            if len(self._document_vector_cache) > self.max_cached_document_vectors:
                # Evaluation/runtime cache only; never persisted. Drop the oldest
                # insertion-order entries rather than allowing unbounded PHI-derived
                # vectors in process memory.
                excess = len(self._document_vector_cache) - self.max_cached_document_vectors
                for old_key in list(self._document_vector_cache)[:excess]:
                    self._document_vector_cache.pop(old_key, None)
            vectors = [
                self._document_vector_cache[(
                    request.scope.patient_key,
                    document.id,
                    self._document_fingerprint(document),
                )]
                for document in documents
            ]
        scored = sorted(
            zip(cosine_scores(query_vector, vectors), documents),
            key=lambda item: item[0],
            reverse=True,
        )
        hits = [
            RetrievalHit(document.id, document.source_kind, float(score), self.embedding_backend.name)
            for score, document in scored[: request.policy.dense_k]
        ]
        result: HybridGraphState = {"dense_hits": hits, "semantic_used": True}
        self._timed(result, "dense", started)
        return result

    def fusion_node(self, state: HybridGraphState) -> dict[str, Any]:
        started = time.perf_counter()
        request = state["request"]
        rankings = [state.get("lexical_hits") or []]
        dense = state.get("dense_hits") or []
        if dense:
            rankings.append(dense)
        fused = reciprocal_rank_fusion(
            rankings,
            k=request.policy.rrf_constant,
            limit=request.policy.fusion_k,
        )
        result: HybridGraphState = {"fused_hits": fused}
        self._timed(result, "fusion", started)
        return result

    def rerank_node(self, state: HybridGraphState) -> dict[str, Any]:
        started = time.perf_counter()
        request = state["request"]
        fused = state.get("fused_hits") or []
        strategy = request.policy.rerank_strategy
        if (
            not request.policy.use_reranker
            or strategy == RerankStrategy.NONE
            or not self.reranker
            or not self.reranker.available
        ):
            result: HybridGraphState = {
                "reranked_hits": fused[: request.policy.rerank_k],
                "blended_scores": {hit.document_id: hit.score for hit in fused[: request.policy.rerank_k]},
            }
            self._timed(result, "rerank", started)
            return result

        by_id = {document.id: document for document in state["scoped_documents"]}
        candidates = [hit for hit in fused if hit.document_id in by_id][: request.policy.rerank_k]
        texts = [by_id[hit.document_id].title + "\n" + by_id[hit.document_id].text for hit in candidates]
        scores = self.reranker.score(request.question, texts)
        reranked_pairs = sorted(zip(scores, candidates), key=lambda item: float(item[0]), reverse=True)
        rerank_rank = {hit.document_id: rank for rank, (_, hit) in enumerate(reranked_pairs, 1)}
        raw_rerank = {hit.document_id: float(score) for score, hit in reranked_pairs}

        if strategy == RerankStrategy.REPLACE:
            hits = [
                RetrievalHit(hit.document_id, hit.source_kind, float(score), self.reranker.name)
                for score, hit in reranked_pairs
            ]
            result = {"reranked_hits": hits, "blended_scores": raw_rerank}
            self._timed(result, "rerank", started)
            return result

        lexical = state.get("lexical_hits") or []
        dense = state.get("dense_hits") or []
        lexical_rank = {hit.document_id: rank for rank, hit in enumerate(lexical, 1)}
        dense_rank = {hit.document_id: rank for rank, hit in enumerate(dense, 1)}
        fusion_rank = {hit.document_id: rank for rank, hit in enumerate(fused, 1)}
        lexical_raw = {hit.document_id: float(hit.score) for hit in lexical}
        dense_raw = {hit.document_id: float(hit.score) for hit in dense}

        if strategy == RerankStrategy.RANK_BLEND:
            blended: list[tuple[float, RetrievalHit]] = []
            for hit in candidates:
                lex_rank = lexical_rank.get(hit.document_id)
                score = (
                    request.policy.lexical_rank_weight * self._rank_component(lex_rank)
                    + request.policy.dense_rank_weight * self._rank_component(dense_rank.get(hit.document_id))
                    + request.policy.fusion_rank_weight * self._rank_component(fusion_rank.get(hit.document_id))
                    + request.policy.rerank_rank_weight * self._rank_component(rerank_rank.get(hit.document_id))
                )
                if lex_rank and lex_rank <= request.policy.lexical_anchor_depth:
                    score += request.policy.lexical_anchor_bonus / float(lex_rank)
                blended.append((score, hit))
        else:
            lex_norm = self._minmax({hit.document_id: lexical_raw[hit.document_id] for hit in candidates if hit.document_id in lexical_raw})
            dense_norm = self._minmax({hit.document_id: dense_raw[hit.document_id] for hit in candidates if hit.document_id in dense_raw})
            rerank_norm = self._minmax({hit.document_id: raw_rerank[hit.document_id] for hit in candidates if hit.document_id in raw_rerank})
            weights = [request.policy.lexical_score_weight, request.policy.dense_score_weight, request.policy.rerank_score_weight]
            if not request.policy.use_reranker:
                weights[2] = 0.0
            total = sum(weights) or 1.0
            weights = [weight / total for weight in weights]
            blended = []
            for hit in candidates:
                score = (
                    weights[0] * lex_norm.get(hit.document_id, 0.0)
                    + weights[1] * dense_norm.get(hit.document_id, 0.0)
                    + weights[2] * rerank_norm.get(hit.document_id, 0.0)
                )
                # Deterministic tiny tie-break keeps stronger fused evidence ahead.
                score += 1e-6 * self._rank_component(fusion_rank.get(hit.document_id))
                blended.append((score, hit))

        blended.sort(key=lambda item: item[0], reverse=True)
        if strategy == RerankStrategy.ANCHORED_SCORE_BLEND:
            protected_ids = [
                hit.document_id for hit in lexical[: request.policy.lexical_anchor_depth]
                if any(candidate.document_id == hit.document_id for candidate in candidates)
            ]
            by_candidate_id = {hit.document_id: hit for hit in candidates}
            score_by_id = {hit.document_id: float(score) for score, hit in blended}
            ordered_ids = protected_ids + [
                hit.document_id for _, hit in blended if hit.document_id not in set(protected_ids)
            ]
            blended = [(score_by_id[doc_id], by_candidate_id[doc_id]) for doc_id in ordered_ids]

        hits = [
            RetrievalHit(hit.document_id, hit.source_kind, float(score), strategy.value)
            for score, hit in blended
        ]
        result = {
            "reranked_hits": hits,
            "blended_scores": {hit.document_id: float(score) for score, hit in blended},
        }
        self._timed(result, "rerank", started)
        return result

    def validate_node(self, state: HybridGraphState) -> dict[str, Any]:
        started = time.perf_counter()
        request = state["request"]
        allowed = set(request.scope.allowed_document_ids)
        scoped_ids = {document.id for document in state["scoped_documents"]}
        candidates = (
            state.get("reranked_hits")
            or state.get("fused_hits")
            or state.get("lexical_hits")
            or []
        )
        final_hits = candidates[: request.policy.final_k]
        unauthorized = [
            hit.document_id
            for hit in final_hits
            if hit.document_id not in allowed or hit.document_id not in scoped_ids
        ]
        if unauthorized:
            raise RetrievalSecurityError(
                "Retriever/reranker produced document IDs outside the authorized scope: "
                + ", ".join(sorted(set(unauthorized)))
            )
        result: HybridGraphState = {"final_hits": final_hits}
        self._timed(result, "validate", started)
        return result

    def _response(self, state: HybridGraphState) -> HybridRetrievalResponse:
        request = state["request"]
        lexical = state.get("lexical_hits") or []
        dense = state.get("dense_hits") or []
        fused = state.get("fused_hits") or []
        reranked = state.get("reranked_hits") or []
        final_hits = state.get("final_hits") or []
        lexical_rank = {hit.document_id: rank for rank, hit in enumerate(lexical, 1)}
        dense_rank = {hit.document_id: rank for rank, hit in enumerate(dense, 1)}
        fusion_score = {hit.document_id: hit.score for hit in fused}
        rerank_score = {hit.document_id: hit.score for hit in reranked}
        lexical_score = {hit.document_id: hit.score for hit in lexical}
        dense_score = {hit.document_id: hit.score for hit in dense}
        blended_score = state.get("blended_scores") or {}
        candidates = [
            CandidateScore(
                document_id=hit.document_id,
                source_kind=hit.source_kind,
                lexical_rank=lexical_rank.get(hit.document_id),
                dense_rank=dense_rank.get(hit.document_id),
                fusion_score=fusion_score.get(hit.document_id),
                rerank_score=rerank_score.get(hit.document_id),
                lexical_score=lexical_score.get(hit.document_id),
                dense_score=dense_score.get(hit.document_id),
                blended_score=blended_score.get(hit.document_id),
                final_rank=rank,
            )
            for rank, hit in enumerate(final_hits, 1)
        ]
        semantic_used = bool(state.get("semantic_used"))
        if request.policy.mode == RetrievalMode.FTS5:
            backend = "patient_scoped_fts5"
        elif not semantic_used and request.policy.rerank_strategy == RerankStrategy.PREFERRED_KIND_RESCUE:
            backend = "patient_scoped_fts5+preferred_kind_gate"
        elif self.reranker and self.reranker.available and request.policy.use_reranker:
            if request.policy.rerank_strategy == RerankStrategy.REPLACE:
                backend = "patient_scoped_fts5+fastembed_rrf+cross_encoder_replace"
            else:
                backend = f"patient_scoped_fts5+fastembed_rrf+{request.policy.rerank_strategy.value}"
        else:
            backend = "patient_scoped_fts5+fastembed_rrf"
        return HybridRetrievalResponse(
            query_id=request.query_id,
            mode=request.policy.mode,
            backend=backend,
            candidates=candidates,
            authorized_document_count=len(state.get("scoped_documents") or []),
            rejected_document_ids=list(state.get("rejected_document_ids") or []),
            patient_leakage_count=0,
            semantic_available=bool(self.embedding_backend and self.embedding_backend.available),
            reranker_available=bool(self.reranker and self.reranker.available),
            semantic_used=semantic_used,
        )

    def run_reference(
        self,
        request: HybridRetrievalRequest,
        documents: list[RetrievalDocument],
    ) -> HybridRetrievalResponse:
        """Dependency-free execution of the same bounded node logic."""
        if external_tracing_enabled():
            raise RetrievalSecurityError(
                "External LangChain/LangSmith tracing is enabled; clinical retrieval refuses to start."
            )
        state: HybridGraphState = {"request": request, "documents": documents, "timings_ms": {}}
        state.update(self.scope_node(state))
        state.update(self.lexical_node(state))
        if request.policy.mode == RetrievalMode.HYBRID_EVAL:
            if self._preferred_kind_gate_satisfied(request, state.get("lexical_hits") or []):
                state["semantic_used"] = False
            else:
                state.update(self.dense_node(state))
                state.update(self.fusion_node(state))
                state.update(self.rerank_node(state))
        state.update(self.validate_node(state))
        return self._response(state)

    def build_langgraph(self):
        """Compile the evaluation workflow with LangGraph when installed."""
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise OptionalDependencyUnavailable(
                "langgraph is not installed; install requirements-ml.txt"
            ) from exc

        if external_tracing_enabled():
            raise RetrievalSecurityError(
                "External LangChain/LangSmith tracing is enabled; clinical retrieval refuses to start."
            )
        builder = StateGraph(HybridGraphState)
        builder.add_node("scope", self.scope_node)
        builder.add_node("lexical", self.lexical_node)
        builder.add_node("dense", self.dense_node)
        builder.add_node("fusion", self.fusion_node)
        builder.add_node("rerank", self.rerank_node)
        builder.add_node("validate", self.validate_node)
        builder.add_edge(START, "scope")
        builder.add_edge("scope", "lexical")

        def after_lexical(state: HybridGraphState) -> str:
            request = state["request"]
            if request.policy.mode != RetrievalMode.HYBRID_EVAL:
                return "validate"
            if self._preferred_kind_gate_satisfied(request, state.get("lexical_hits") or []):
                return "validate"
            return "dense"

        builder.add_conditional_edges("lexical", after_lexical, {"dense": "dense", "validate": "validate"})
        builder.add_edge("dense", "fusion")
        builder.add_edge("fusion", "rerank")
        builder.add_edge("rerank", "validate")
        builder.add_edge("validate", END)
        return builder.compile()

    def run_langgraph(
        self,
        request: HybridRetrievalRequest,
        documents: list[RetrievalDocument],
    ) -> HybridRetrievalResponse:
        graph = self.build_langgraph()
        result = graph.invoke({"request": request, "documents": documents, "timings_ms": {}})
        return self._response(result)
