"""Benchmark helpers for local decision-model retrieval experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Any, Callable

from .decision_rag import DecisionQueryResult, DecisionRetrievalPolicy, PatientDecisionRetriever
from .hybrid_benchmark import HardRetrievalQuestion
from .retrieval_eval import FTS5Retriever, RetrievalDocument

DECISION_BENCHMARK_REVISION = "decision-retrieval-benchmark-v1"


@dataclass(frozen=True)
class DecisionBenchmarkSummary:
    backend: str
    supported_questions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    route_accuracy_at_1: float | None
    route_accuracy_at_3: float | None
    patient_leakage_count: int
    mean_latency_ms: float
    mean_decision_ms: float
    mean_dense_ms: float
    mean_decision_calls: float

    def public(self) -> dict[str, Any]:
        return asdict(self)


def stratified_questions(questions: list[HardRetrievalQuestion], max_questions: int | None, seed: int = 42) -> list[HardRetrievalQuestion]:
    supported = [q for q in questions if q.answerability == "supported" and q.expected_source_kind]
    if not max_questions or max_questions >= len(supported):
        return supported
    by_category: dict[str, list[HardRetrievalQuestion]] = {}
    for q in supported:
        by_category.setdefault(q.category, []).append(q)
    rng = random.Random(seed)
    for rows in by_category.values():
        rng.shuffle(rows)
    categories = sorted(by_category)
    chosen: list[HardRetrievalQuestion] = []
    cursor = 0
    while len(chosen) < max_questions and categories:
        category = categories[cursor % len(categories)]
        rows = by_category[category]
        if rows:
            chosen.append(rows.pop())
        else:
            categories.remove(category)
            if not categories:
                break
            cursor -= 1
        cursor += 1
    return chosen


def _dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances[:k]))


def evaluate_decision_strategy(
    retriever: PatientDecisionRetriever,
    documents: list[RetrievalDocument],
    questions: list[HardRetrievalQuestion],
    strategy: str,
    policy: DecisionRetrievalPolicy,
) -> DecisionBenchmarkSummary:
    by_patient: dict[str, list[RetrievalDocument]] = {}
    for doc in documents:
        by_patient.setdefault(doc.patient_key, []).append(doc)
    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal = 0.0
    ndcg = 0.0
    route1 = 0
    route3 = 0
    route_questions = 0
    leakage = 0
    total_ms = decision_ms = dense_ms = calls = 0.0
    for q in questions:
        scoped = list(by_patient.get(q.patient_key) or [])
        started = time.perf_counter()
        if strategy == "decision_rerank":
            result = retriever.rerank_union(q.question, scoped, policy)
        elif strategy == "decision_navigation":
            result = retriever.navigate(q.question, scoped, DecisionRetrievalPolicy(**{**asdict(policy), "dense_rescue": False}))
        elif strategy == "decision_navigation_dense":
            result = retriever.navigate(q.question, scoped, DecisionRetrievalPolicy(**{**asdict(policy), "dense_rescue": True}))
        else:
            raise ValueError(f"unsupported decision strategy: {strategy}")
        total_ms += (time.perf_counter() - started) * 1000.0
        decision_ms += result.decision_ms
        dense_ms += result.dense_ms
        calls += result.decision_calls
        allowed = {doc.id for doc in scoped}
        if any(hit.document_id not in allowed for hit in result.hits):
            leakage += 1
        target_rank = None
        relevances: list[int] = []
        for rank, hit in enumerate(result.hits, 1):
            relevant = int(hit.source_kind == q.expected_source_kind)
            relevances.append(relevant)
            if relevant and target_rank is None:
                target_rank = rank
        if target_rank:
            reciprocal += 1.0 / target_rank
            for cutoff in hits_at:
                if target_rank <= cutoff:
                    hits_at[cutoff] += 1
        ideal_relevant = min(sum(1 for doc in scoped if doc.source_kind == q.expected_source_kind), 5)
        ideal = _dcg([1] * ideal_relevant, 5)
        ndcg += (_dcg(relevances, 5) / ideal) if ideal else 0.0
        if result.route_probabilities:
            route_questions += 1
            ordered = [k for k, _ in sorted(result.route_probabilities.items(), key=lambda x: x[1], reverse=True)]
            if ordered and ordered[0] == q.expected_source_kind:
                route1 += 1
            if q.expected_source_kind in ordered[:3]:
                route3 += 1
    n = max(1, len(questions))
    return DecisionBenchmarkSummary(
        backend=strategy,
        supported_questions=len(questions),
        recall_at_1=round(hits_at[1] / n, 4),
        recall_at_3=round(hits_at[3] / n, 4),
        recall_at_5=round(hits_at[5] / n, 4),
        mrr=round(reciprocal / n, 4),
        ndcg_at_5=round(ndcg / n, 4),
        route_accuracy_at_1=round(route1 / route_questions, 4) if route_questions else None,
        route_accuracy_at_3=round(route3 / route_questions, 4) if route_questions else None,
        patient_leakage_count=leakage,
        mean_latency_ms=round(total_ms / n, 3),
        mean_decision_ms=round(decision_ms / n, 3),
        mean_dense_ms=round(dense_ms / n, 3),
        mean_decision_calls=round(calls / n, 3),
    )
