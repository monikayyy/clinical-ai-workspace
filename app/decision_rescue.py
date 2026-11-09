"""Confidence-gated local semantic rescue for patient-scoped retrieval.

This module is evaluation-only. It never authorizes patients itself; callers
must pass documents for exactly one already-authorized patient. Normalized FTS5
remains the first-stage retriever. Dense retrieval and the local decision model
are invoked only when the lexical evidence gate is not satisfied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import re
import time
from typing import Any

from .decision_models import DecisionBackend
from .decision_rag import DecisionRetrievalSecurityError, PatientDecisionRetriever, _scope_patient
from .embedding_backends import EmbeddingBackend, cosine_scores
from .retrieval_eval import FTS5Retriever, RetrievalDocument, RetrievalHit, query_profile, reciprocal_rank_fusion

DECISION_RESCUE_REVISION = "decision-rescue-eval-v1"


@dataclass(frozen=True)
class DecisionRescuePolicy:
    lexical_k: int = 12
    gate_depth: int = 5
    dense_k: int = 12
    candidate_k: int = 16
    final_k: int = 5
    use_dense: bool = True
    use_decision: bool = True
    decision_min_top_probability: float = 0.0
    decision_min_margin: float = 0.0
    fallback_rrf_constant: int = 60


@dataclass(frozen=True)
class LexicalConfidence:
    strong: bool
    preferred_source_kinds: tuple[str, ...]
    reason: str
    gate_depth: int
    lexical_hit_count: int
    preferred_kind_rank: int | None
    top_score: float | None
    second_score: float | None
    score_margin: float | None

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionRescueResult:
    backend: str
    hits: tuple[RetrievalHit, ...]
    confidence: LexicalConfidence
    semantic_used: bool
    dense_used: bool
    decision_used: bool
    decision_accepted: bool
    decision_calls: int
    lexical_ms: float
    dense_ms: float
    decision_ms: float
    candidate_count: int
    decision_top_probability: float | None = None
    decision_margin: float | None = None
    patient_leakage_count: int = 0

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["hits"] = [asdict(hit) for hit in self.hits]
        data["confidence"] = self.confidence.public()
        return data


def rescue_query_profile(question: str) -> dict[str, Any]:
    """Deterministic routing vocabulary used only by the confidence gate.

    This intentionally does not alter FTS query terms. It broadens source-type
    recognition for user phrasings that may have little lexical overlap with a
    chart, so a low-confidence lexical result can be sent to local semantic
    rescue. It never changes patient scope or asserts a clinical fact.
    """
    base = query_profile(question)
    preferred = set(base.get("preferred_source_kinds") or ())
    if preferred:
        return {**base, "preferred_source_kinds": preferred}

    q = question.casefold()
    observation_patterns = (
        r"\bsugar\s+control\b",
        r"\bthree[- ]month\s+glucose\b",
        r"\bbody[- ]?size\s+index\b",
        r"\bweight[- ]?for[- ]?height\b",
        r"\bnose\b",
        r"\ballergy\s+symptoms?\b",
        r"\bhow\s+often\b.*\b(?:migraine|headache|heartburn|reflux|nasal|symptoms?)\b",
    )
    if any(re.search(pattern, q) for pattern in observation_patterns):
        return {
            "kind": "measurement",
            "preferred_source_kinds": {"observation_report"},
            "expansion": list(base.get("expansion") or ()),
        }
    return {**base, "preferred_source_kinds": preferred}


def lexical_confidence(
    question: str,
    lexical_hits: list[RetrievalHit],
    policy: DecisionRescuePolicy,
) -> LexicalConfidence:
    profile = rescue_query_profile(question)
    preferred = tuple(sorted(profile.get("preferred_source_kinds") or ()))
    depth = min(max(1, policy.gate_depth), len(lexical_hits))
    preferred_rank = None
    if preferred:
        preferred_set = set(preferred)
        for rank, hit in enumerate(lexical_hits[:depth], 1):
            if hit.source_kind in preferred_set:
                preferred_rank = rank
                break

    top = float(lexical_hits[0].score) if lexical_hits else None
    second = float(lexical_hits[1].score) if len(lexical_hits) > 1 else None
    margin = (top - second) if top is not None and second is not None else None

    if not preferred:
        strong = False
        reason = "no_deterministic_source_type"
    elif preferred_rank is not None:
        strong = True
        reason = "preferred_source_type_present"
    else:
        strong = False
        reason = "preferred_source_type_missing"

    return LexicalConfidence(
        strong=strong,
        preferred_source_kinds=preferred,
        reason=reason,
        gate_depth=policy.gate_depth,
        lexical_hit_count=len(lexical_hits),
        preferred_kind_rank=preferred_rank,
        top_score=top,
        second_score=second,
        score_margin=margin,
    )


def _unique_documents_from_hits(
    hits: list[RetrievalHit],
    by_id: dict[str, RetrievalDocument],
    output: list[RetrievalDocument],
    limit: int,
) -> None:
    seen = {doc.id for doc in output}
    for hit in hits:
        doc = by_id.get(hit.document_id)
        if doc and doc.id not in seen:
            output.append(doc)
            seen.add(doc.id)
            if len(output) >= limit:
                return


class ConfidenceGatedDecisionRetriever:
    """FTS-first retrieval with local dense/decision fallback only on uncertainty."""

    def __init__(
        self,
        decision_backend: DecisionBackend | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ):
        self.decision = decision_backend
        self.embedding = embedding_backend
        self.fts = FTS5Retriever()
        self._decision_retriever = (
            PatientDecisionRetriever(decision_backend, embedding_backend)
            if decision_backend is not None
            else None
        )
        # Patient/document/hash scoped RAM-only cache. Nothing is persisted.
        self._document_vectors: dict[tuple[str, str, str], Any] = {}

    @staticmethod
    def _document_fingerprint(document: RetrievalDocument) -> str:
        return hashlib.sha256((document.title + "\n" + document.text).encode("utf-8")).hexdigest()

    def _dense(self, question: str, documents: list[RetrievalDocument]) -> tuple[list[RetrievalHit], float]:
        if not self.embedding or not self.embedding.available or not documents:
            return [], 0.0
        started = time.perf_counter()
        query_vector = self.embedding.embed_query(question)
        patient = _scope_patient(documents)
        missing_keys: list[tuple[str, str, str]] = []
        missing_texts: list[str] = []
        keys: list[tuple[str, str, str]] = []
        for doc in documents:
            key = (patient, doc.id, self._document_fingerprint(doc))
            keys.append(key)
            if key not in self._document_vectors:
                missing_keys.append(key)
                missing_texts.append(doc.title + "\n" + doc.text)
        if missing_texts:
            vectors = self.embedding.embed_documents(missing_texts)
            for key, vector in zip(missing_keys, vectors):
                self._document_vectors[key] = vector
        vectors = [self._document_vectors[key] for key in keys]
        scored = sorted(
            zip(cosine_scores(query_vector, vectors), documents),
            key=lambda item: item[0],
            reverse=True,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        return [
            RetrievalHit(doc.id, doc.source_kind, float(score), self.embedding.name)
            for score, doc in scored
        ], elapsed

    @staticmethod
    def _decision_quality(probabilities: dict[str, float]) -> tuple[float | None, float | None]:
        if not probabilities:
            return None, None
        values = sorted((float(value) for value in probabilities.values()), reverse=True)
        top = values[0]
        second = values[1] if len(values) > 1 else 0.0
        return top, top - second

    def retrieve(
        self,
        question: str,
        documents: list[RetrievalDocument],
        policy: DecisionRescuePolicy = DecisionRescuePolicy(),
    ) -> DecisionRescueResult:
        _scope_patient(documents)
        if policy.final_k < 1 or policy.candidate_k < policy.final_k:
            raise ValueError("candidate_k must be >= final_k >= 1")
        if policy.use_dense and (not self.embedding or not self.embedding.available):
            raise RuntimeError("dense rescue requested but the local embedding backend is unavailable")
        if policy.use_decision and (
            not self.decision or not getattr(self.decision, "available", False) or not self._decision_retriever
        ):
            raise RuntimeError("decision rescue requested but the local decision backend is unavailable")

        lexical_started = time.perf_counter()
        lexical = self.fts.rank(question, documents, limit=max(policy.lexical_k, policy.gate_depth, policy.final_k))
        lexical_ms = (time.perf_counter() - lexical_started) * 1000.0
        confidence = lexical_confidence(question, lexical, policy)

        # The normal path is intentionally identical to FTS5. No patient text is
        # passed to embeddings or the decision model when the lexical gate passes.
        if confidence.strong:
            return DecisionRescueResult(
                backend="patient_scoped_fts5+confidence_gate",
                hits=tuple(lexical[: policy.final_k]),
                confidence=confidence,
                semantic_used=False,
                dense_used=False,
                decision_used=False,
                decision_accepted=False,
                decision_calls=0,
                lexical_ms=round(lexical_ms, 3),
                dense_ms=0.0,
                decision_ms=0.0,
                candidate_count=min(len(lexical), policy.final_k),
            )

        by_id = {doc.id: doc for doc in documents}
        dense: list[RetrievalHit] = []
        dense_ms = 0.0
        if policy.use_dense:
            dense, dense_ms = self._dense(question, documents)

        candidates: list[RetrievalDocument] = []
        _unique_documents_from_hits(lexical[: policy.lexical_k], by_id, candidates, policy.candidate_k)
        _unique_documents_from_hits(dense[: policy.dense_k], by_id, candidates, policy.candidate_k)

        # If FTS produced no candidates and dense rescue is disabled, keep a
        # bounded patient-local candidate set for Arc rather than returning an
        # empty result. This is evaluation-only and never changes authorization.
        if policy.use_decision and not candidates:
            candidates.extend(documents[: policy.candidate_k])

        decision_used = False
        decision_accepted = False
        decision_calls = 0
        decision_ms = 0.0
        top_probability = None
        margin = None
        ranked: list[RetrievalHit]

        if policy.use_decision and candidates:
            ranked, probabilities, decision_calls, decision_ms = self._decision_retriever.rank_candidates(
                question,
                candidates[: policy.candidate_k],
            )
            decision_used = True
            top_probability, margin = self._decision_quality(probabilities)
            decision_accepted = bool(
                ranked
                and (top_probability or 0.0) >= policy.decision_min_top_probability
                and (margin or 0.0) >= policy.decision_min_margin
            )
        else:
            ranked = []

        if not decision_accepted:
            # If Arc is unavailable/uncertain, preserve a deterministic lexical+
            # dense union rather than trusting a weak model decision.
            rankings: list[list[RetrievalHit]] = [lexical[: policy.lexical_k]]
            if dense:
                rankings.append(dense[: policy.dense_k])
            ranked = reciprocal_rank_fusion(
                rankings,
                k=policy.fallback_rrf_constant,
                limit=policy.candidate_k,
            )

        allowed = set(by_id)
        if any(hit.document_id not in allowed for hit in ranked):
            raise DecisionRetrievalSecurityError("semantic rescue returned an out-of-scope document")

        backend_parts = ["confidence_gated_rescue"]
        if policy.use_dense:
            backend_parts.append("dense")
        if policy.use_decision:
            backend_parts.append("arc" if decision_accepted else "arc_fallback")
        return DecisionRescueResult(
            backend="+".join(backend_parts),
            hits=tuple(ranked[: policy.final_k]),
            confidence=confidence,
            semantic_used=bool(policy.use_dense or policy.use_decision),
            dense_used=bool(policy.use_dense),
            decision_used=decision_used,
            decision_accepted=decision_accepted,
            decision_calls=decision_calls,
            lexical_ms=round(lexical_ms, 3),
            dense_ms=round(dense_ms, 3),
            decision_ms=round(decision_ms, 3),
            candidate_count=len(candidates),
            decision_top_probability=round(top_probability, 6) if top_probability is not None else None,
            decision_margin=round(margin, 6) if margin is not None else None,
        )
