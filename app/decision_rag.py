"""Patient-scoped DecisionRAG-style retrieval evaluation.

This module never authorizes a patient. It accepts an already-authorized source
set and fails closed if mixed patient content is supplied. The hierarchy is
built from clinical source kinds and local document metadata, then a local
DecisionBackend jointly compares branches/candidates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
import time
from typing import Any, Iterable

from .decision_models import DECISION_RAG_REVISION, DecisionBackend
from .embedding_backends import EmbeddingBackend, cosine_scores
from .retrieval_eval import FTS5Retriever, RetrievalDocument, RetrievalHit

MAX_DECISION_OPTIONS = 16


_SOURCE_KIND_DESCRIPTIONS = {
    "encounter_summary": "Primary-care encounter summary with active conditions, assessment, and visit context.",
    "observation_report": "Laboratory, vital-sign, or clinical measurement report with result values.",
    "medication_reconciliation": "Medication reconciliation with documented/current drug therapy.",
    "outside_consult": "Outside specialist consultation or external report update.",
    "longitudinal_summary": "Longitudinal summary spanning multiple visits and clinical changes over time.",
    "simulated_scanned_note": "Scanned or OCR-like clinical note containing narrative chart information.",
    "note": "General clinical note, follow-up note, or chart narrative.",
}


class DecisionRetrievalSecurityError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecisionRetrievalPolicy:
    branch_k: int = 3
    lexical_k: int = 8
    dense_k: int = 8
    candidate_k: int = 12
    final_k: int = 5
    dense_rescue: bool = True
    lexical_anchor_depth: int = 1


@dataclass(frozen=True)
class DecisionQueryResult:
    backend: str
    hits: tuple[RetrievalHit, ...]
    route_probabilities: dict[str, float]
    candidate_probabilities: dict[str, float]
    decision_calls: int
    decision_ms: float
    dense_ms: float
    lexical_ms: float
    patient_leakage_count: int = 0

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["hits"] = [asdict(hit) for hit in self.hits]
        return data


def _source_kind_description(kind: str) -> str:
    return _SOURCE_KIND_DESCRIPTIONS.get(kind, f"Clinical source of type {kind.replace('_', ' ')}.")


def _document_date(text: str) -> str | None:
    match = re.search(r"(?im)^Document date:\s*(\d{4}-\d{2}-\d{2})\s*$", text)
    return match.group(1) if match else None


def _document_option(document: RetrievalDocument, max_chars: int = 1800) -> str:
    date = _document_date(document.text)
    head = f"{document.source_kind.replace('_', ' ')}; {document.title}"
    if date:
        head += f"; date {date}"
    body = document.text.strip().replace("\x00", " ")
    if len(body) > max_chars:
        body = body[:max_chars] + " …"
    return head + "\n" + body


def _scope_patient(documents: list[RetrievalDocument]) -> str:
    patients = {doc.patient_key for doc in documents}
    if len(patients) != 1:
        raise DecisionRetrievalSecurityError("Decision retrieval requires exactly one authorized patient scope")
    return next(iter(patients))


def _safe_choice(
    backend: DecisionBackend,
    state: str,
    options: dict[str, str],
    instruction: str,
) -> dict[str, float]:
    if len(options) < 2:
        if len(options) == 1:
            return {next(iter(options)): 1.0}
        return {}
    if len(options) > MAX_DECISION_OPTIONS:
        raise ValueError(f"decision choice exceeds {MAX_DECISION_OPTIONS} options")
    return backend.choose(state, options, instruction=instruction)


class PatientDecisionRetriever:
    def __init__(
        self,
        decision_backend: DecisionBackend,
        embedding_backend: EmbeddingBackend | None = None,
    ):
        self.decision = decision_backend
        self.embedding = embedding_backend
        self.fts = FTS5Retriever()

    def _route(self, question: str, documents: list[RetrievalDocument]) -> tuple[dict[str, float], int, float]:
        kinds = sorted({doc.source_kind for doc in documents})
        options = {f"b{index}": f"{kind}: {_source_kind_description(kind)}" for index, kind in enumerate(kinds)}
        key_to_kind = {f"b{index}": kind for index, kind in enumerate(kinds)}
        if len(options) == 1:
            return {next(iter(key_to_kind.values())): 1.0}, 0, 0.0
        started = time.perf_counter()
        raw = _safe_choice(
            self.decision,
            question,
            options,
            "Which clinical evidence branch is most likely to contain information needed to answer this question?",
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        return {key_to_kind[key]: prob for key, prob in raw.items()}, 1, elapsed

    def _dense(self, question: str, documents: list[RetrievalDocument]) -> tuple[list[RetrievalHit], float]:
        if not self.embedding or not self.embedding.available or not documents:
            return [], 0.0
        started = time.perf_counter()
        query_vector = self.embedding.embed_query(question)
        vectors = self.embedding.embed_documents([doc.title + "\n" + doc.text for doc in documents])
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

    def _joint_rank(
        self,
        question: str,
        documents: list[RetrievalDocument],
    ) -> tuple[list[RetrievalHit], dict[str, float], int, float]:
        if not documents:
            return [], {}, 0, 0.0
        # Decision heads accept at most 16 options. Candidate generation must be
        # bounded before this point.
        documents = documents[:MAX_DECISION_OPTIONS]
        options = {f"c{index}": _document_option(doc) for index, doc in enumerate(documents)}
        key_to_doc = {f"c{index}": doc for index, doc in enumerate(documents)}
        if len(options) == 1:
            doc = next(iter(key_to_doc.values()))
            return [RetrievalHit(doc.id, doc.source_kind, 1.0, self.decision.name)], {doc.id: 1.0}, 0, 0.0
        started = time.perf_counter()
        raw = _safe_choice(
            self.decision,
            question,
            options,
            "Rank the candidate clinical sources by which one most directly contains evidence needed to answer the question. Choose the best source.",
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        scored = sorted(
            ((float(prob), key_to_doc[key]) for key, prob in raw.items()),
            key=lambda item: item[0],
            reverse=True,
        )
        probabilities = {doc.id: float(score) for score, doc in scored}
        return [
            RetrievalHit(doc.id, doc.source_kind, float(score), self.decision.name)
            for score, doc in scored
        ], probabilities, 1, elapsed

    def rank_candidates(
        self,
        question: str,
        documents: list[RetrievalDocument],
    ) -> tuple[list[RetrievalHit], dict[str, float], int, float]:
        """Jointly rank an already-authorized bounded candidate set."""
        _scope_patient(documents)
        return self._joint_rank(question, documents)

    def navigate(
        self,
        question: str,
        documents: list[RetrievalDocument],
        policy: DecisionRetrievalPolicy = DecisionRetrievalPolicy(),
    ) -> DecisionQueryResult:
        _scope_patient(documents)
        route, calls, decision_ms = self._route(question, documents)
        selected_kinds = [kind for kind, _ in sorted(route.items(), key=lambda x: x[1], reverse=True)[: policy.branch_k]]
        selected = [doc for doc in documents if doc.source_kind in set(selected_kinds)]
        # If branch selection is very narrow, retain enough alternatives for the
        # joint decision rather than irreversibly pruning to zero/one documents.
        if len(selected) < min(3, len(documents)):
            selected = list(documents)
        lexical_started = time.perf_counter()
        lexical = self.fts.rank(question, selected, limit=policy.candidate_k)
        lexical_ms = (time.perf_counter() - lexical_started) * 1000.0
        by_id = {doc.id: doc for doc in documents}
        candidates: list[RetrievalDocument] = [by_id[hit.document_id] for hit in lexical if hit.document_id in by_id]
        dense_ms = 0.0
        if policy.dense_rescue and self.embedding and self.embedding.available:
            dense, dense_ms = self._dense(question, documents)
            for hit in dense[: policy.dense_k]:
                doc = by_id.get(hit.document_id)
                if doc and all(existing.id != doc.id for existing in candidates):
                    candidates.append(doc)
        # Preserve selected-branch documents that lexical retrieval did not score.
        for doc in selected:
            if all(existing.id != doc.id for existing in candidates):
                candidates.append(doc)
        candidates = candidates[: policy.candidate_k]
        ranked, probs, rank_calls, rank_ms = self._joint_rank(question, candidates)
        calls += rank_calls
        decision_ms += rank_ms
        return DecisionQueryResult(
            backend="decision_rag_navigation_dense" if policy.dense_rescue and self.embedding and self.embedding.available else "decision_rag_navigation",
            hits=tuple(ranked[: policy.final_k]),
            route_probabilities=route,
            candidate_probabilities=probs,
            decision_calls=calls,
            decision_ms=round(decision_ms, 3),
            dense_ms=round(dense_ms, 3),
            lexical_ms=round(lexical_ms, 3),
        )

    def rerank_union(
        self,
        question: str,
        documents: list[RetrievalDocument],
        policy: DecisionRetrievalPolicy = DecisionRetrievalPolicy(),
    ) -> DecisionQueryResult:
        _scope_patient(documents)
        lexical_started = time.perf_counter()
        lexical = self.fts.rank(question, documents, limit=policy.lexical_k)
        lexical_ms = (time.perf_counter() - lexical_started) * 1000.0
        dense, dense_ms = self._dense(question, documents)
        by_id = {doc.id: doc for doc in documents}
        candidates: list[RetrievalDocument] = []
        for hit in lexical[: policy.lexical_k] + dense[: policy.dense_k]:
            doc = by_id.get(hit.document_id)
            if doc and all(existing.id != doc.id for existing in candidates):
                candidates.append(doc)
        candidates = candidates[: policy.candidate_k]
        ranked, probs, calls, decision_ms = self._joint_rank(question, candidates)
        # Protect a very strong lexical leader when requested; this prevents the
        # decision model from destroying an exact lexical match while still
        # allowing it to reorder the ambiguous remainder.
        if policy.lexical_anchor_depth > 0 and lexical:
            protected = [hit.document_id for hit in lexical[: policy.lexical_anchor_depth]]
            by_ranked = {hit.document_id: hit for hit in ranked}
            anchored = [by_ranked[doc_id] for doc_id in protected if doc_id in by_ranked]
            anchored_ids = {hit.document_id for hit in anchored}
            anchored.extend(hit for hit in ranked if hit.document_id not in anchored_ids)
            ranked = anchored
        return DecisionQueryResult(
            backend="decision_joint_rerank",
            hits=tuple(ranked[: policy.final_k]),
            route_probabilities={},
            candidate_probabilities=probs,
            decision_calls=calls,
            decision_ms=round(decision_ms, 3),
            dense_ms=round(dense_ms, 3),
            lexical_ms=round(lexical_ms, 3),
        )
