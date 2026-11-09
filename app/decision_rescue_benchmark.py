"""Full hard-corpus benchmark for confidence-gated local semantic rescue."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
import time
from typing import Any

from .decision_rescue import ConfidenceGatedDecisionRetriever, DecisionRescuePolicy, lexical_confidence
from .hybrid_benchmark import HardRetrievalQuestion
from .retrieval_eval import FTS5Retriever, RetrievalDocument

DECISION_RESCUE_BENCHMARK_REVISION = "decision-rescue-benchmark-v1"


@dataclass(frozen=True)
class GateSummary:
    supported_questions: int
    rescue_queries: int
    rescue_fraction: float
    fts_failures: int
    fts_failures_triggered: int
    fts_failure_trigger_rate: float
    fts_successes_triggered: int
    fts_success_trigger_rate: float
    patient_leakage_count: int
    by_category: dict[str, dict[str, int | float]]

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RescueBenchmarkSummary:
    backend: str
    supported_questions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    patient_leakage_count: int
    rescue_queries: int
    rescue_fraction: float
    dense_queries: int
    decision_queries: int
    decision_accepted_queries: int
    rescued_fts_failures: int
    damaged_fts_successes: int
    remaining_fts_failures: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_dense_ms_on_rescue: float
    mean_decision_ms_on_rescue: float
    mean_decision_calls_on_rescue: float
    by_category: dict[str, dict[str, int | float]]

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances[:k]))


def _target_rank(hits, expected_source_kind: str) -> int | None:
    for index, hit in enumerate(hits, 1):
        if hit.source_kind == expected_source_kind:
            return index
    return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * percentile
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def evaluate_gate(
    documents: list[RetrievalDocument],
    questions: list[HardRetrievalQuestion],
    policy: DecisionRescuePolicy,
) -> GateSummary:
    by_patient: dict[str, list[RetrievalDocument]] = {}
    for doc in documents:
        by_patient.setdefault(doc.patient_key, []).append(doc)
    fts = FTS5Retriever()
    supported = [q for q in questions if q.answerability == "supported" and q.expected_source_kind]
    rescue_queries = fts_failures = failure_triggered = success_triggered = leakage = 0
    by_category_raw: dict[str, dict[str, int]] = {}

    for q in supported:
        scoped = list(by_patient.get(q.patient_key) or [])
        if any(doc.patient_key != q.patient_key for doc in scoped):
            leakage += 1
            continue
        lexical = fts.rank(q.question, scoped, limit=max(policy.lexical_k, policy.gate_depth, policy.final_k))
        baseline_success = _target_rank(lexical[: policy.final_k], q.expected_source_kind) is not None
        confidence = lexical_confidence(q.question, lexical, policy)
        rescue = not confidence.strong
        rescue_queries += int(rescue)
        fts_failures += int(not baseline_success)
        failure_triggered += int((not baseline_success) and rescue)
        success_triggered += int(baseline_success and rescue)
        row = by_category_raw.setdefault(q.category, {
            "questions": 0,
            "rescue_queries": 0,
            "fts_failures": 0,
            "fts_failures_triggered": 0,
            "fts_successes_triggered": 0,
        })
        row["questions"] += 1
        row["rescue_queries"] += int(rescue)
        row["fts_failures"] += int(not baseline_success)
        row["fts_failures_triggered"] += int((not baseline_success) and rescue)
        row["fts_successes_triggered"] += int(baseline_success and rescue)

    n = max(1, len(supported))
    successes = max(1, len(supported) - fts_failures)
    return GateSummary(
        supported_questions=len(supported),
        rescue_queries=rescue_queries,
        rescue_fraction=round(rescue_queries / n, 4),
        fts_failures=fts_failures,
        fts_failures_triggered=failure_triggered,
        fts_failure_trigger_rate=round(failure_triggered / max(1, fts_failures), 4),
        fts_successes_triggered=success_triggered,
        fts_success_trigger_rate=round(success_triggered / successes, 4),
        patient_leakage_count=leakage,
        by_category=by_category_raw,
    )


def evaluate_rescue_strategy(
    retriever: ConfidenceGatedDecisionRetriever,
    documents: list[RetrievalDocument],
    questions: list[HardRetrievalQuestion],
    policy: DecisionRescuePolicy,
    backend: str,
) -> RescueBenchmarkSummary:
    by_patient: dict[str, list[RetrievalDocument]] = {}
    for doc in documents:
        by_patient.setdefault(doc.patient_key, []).append(doc)
    fts = FTS5Retriever()
    supported = [q for q in questions if q.answerability == "supported" and q.expected_source_kind]
    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal = ndcg = 0.0
    leakage = rescue_queries = dense_queries = decision_queries = decision_accepted = 0
    rescued_failures = damaged_successes = remaining_failures = 0
    latencies: list[float] = []
    rescue_dense_ms: list[float] = []
    rescue_decision_ms: list[float] = []
    rescue_decision_calls: list[float] = []
    by_category_raw: dict[str, dict[str, int | float]] = {}

    for q in supported:
        scoped = list(by_patient.get(q.patient_key) or [])
        allowed = {doc.id for doc in scoped}
        baseline = fts.rank(q.question, scoped, limit=max(policy.lexical_k, policy.final_k))
        baseline_success = _target_rank(baseline[: policy.final_k], q.expected_source_kind) is not None

        started = time.perf_counter()
        result = retriever.retrieve(q.question, scoped, policy)
        latencies.append((time.perf_counter() - started) * 1000.0)
        if any(hit.document_id not in allowed for hit in result.hits):
            leakage += 1

        result_success = _target_rank(result.hits[: policy.final_k], q.expected_source_kind) is not None
        rescue_queries += int(result.semantic_used)
        dense_queries += int(result.dense_used)
        decision_queries += int(result.decision_used)
        decision_accepted += int(result.decision_accepted)
        if result.semantic_used:
            rescue_dense_ms.append(result.dense_ms)
            rescue_decision_ms.append(result.decision_ms)
            rescue_decision_calls.append(float(result.decision_calls))
        rescued_failures += int((not baseline_success) and result_success)
        damaged_successes += int(baseline_success and (not result_success))
        remaining_failures += int((not baseline_success) and (not result_success))

        rank = _target_rank(result.hits, q.expected_source_kind)
        relevances = [int(hit.source_kind == q.expected_source_kind) for hit in result.hits]
        if rank is not None:
            reciprocal += 1.0 / rank
            for cutoff in hits_at:
                if rank <= cutoff:
                    hits_at[cutoff] += 1
        ideal_relevant = min(sum(1 for doc in scoped if doc.source_kind == q.expected_source_kind), 5)
        ideal = _dcg([1] * ideal_relevant, 5)
        ndcg += (_dcg(relevances, 5) / ideal) if ideal else 0.0

        row = by_category_raw.setdefault(q.category, {
            "questions": 0,
            "rescue_queries": 0,
            "rescued_fts_failures": 0,
            "damaged_fts_successes": 0,
            "hits_at_5": 0,
            "reciprocal_sum": 0.0,
        })
        row["questions"] = int(row["questions"]) + 1
        row["rescue_queries"] = int(row["rescue_queries"]) + int(result.semantic_used)
        row["rescued_fts_failures"] = int(row["rescued_fts_failures"]) + int((not baseline_success) and result_success)
        row["damaged_fts_successes"] = int(row["damaged_fts_successes"]) + int(baseline_success and (not result_success))
        if rank is not None:
            row["reciprocal_sum"] = float(row["reciprocal_sum"]) + 1.0 / rank
            if rank <= 5:
                row["hits_at_5"] = int(row["hits_at_5"]) + 1

    n = max(1, len(supported))
    by_category: dict[str, dict[str, int | float]] = {}
    for category, row in by_category_raw.items():
        count = max(1, int(row["questions"]))
        by_category[category] = {
            "questions": int(row["questions"]),
            "rescue_queries": int(row["rescue_queries"]),
            "rescued_fts_failures": int(row["rescued_fts_failures"]),
            "damaged_fts_successes": int(row["damaged_fts_successes"]),
            "recall_at_5": round(int(row["hits_at_5"]) / count, 4),
            "mrr": round(float(row["reciprocal_sum"]) / count, 4),
        }

    return RescueBenchmarkSummary(
        backend=backend,
        supported_questions=len(supported),
        recall_at_1=round(hits_at[1] / n, 4),
        recall_at_3=round(hits_at[3] / n, 4),
        recall_at_5=round(hits_at[5] / n, 4),
        mrr=round(reciprocal / n, 4),
        ndcg_at_5=round(ndcg / n, 4),
        patient_leakage_count=leakage,
        rescue_queries=rescue_queries,
        rescue_fraction=round(rescue_queries / n, 4),
        dense_queries=dense_queries,
        decision_queries=decision_queries,
        decision_accepted_queries=decision_accepted,
        rescued_fts_failures=rescued_failures,
        damaged_fts_successes=damaged_successes,
        remaining_fts_failures=remaining_failures,
        mean_latency_ms=round(statistics.fmean(latencies) if latencies else 0.0, 3),
        p50_latency_ms=round(_percentile(latencies, 0.50), 3),
        p95_latency_ms=round(_percentile(latencies, 0.95), 3),
        mean_dense_ms_on_rescue=round(statistics.fmean(rescue_dense_ms) if rescue_dense_ms else 0.0, 3),
        mean_decision_ms_on_rescue=round(statistics.fmean(rescue_decision_ms) if rescue_decision_ms else 0.0, 3),
        mean_decision_calls_on_rescue=round(statistics.fmean(rescue_decision_calls) if rescue_decision_calls else 0.0, 3),
        by_category=by_category,
    )


def choose_rescue_winner(baseline: dict[str, Any], strategies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    safe = []
    for name, row in strategies.items():
        if int(row.get("patient_leakage_count") or 0) != 0:
            continue
        if int(row.get("damaged_fts_successes") or 0) != 0:
            continue
        if float(row.get("recall_at_5") or 0.0) < float(baseline.get("recall_at_5") or 0.0):
            continue
        if float(row.get("mrr") or 0.0) < float(baseline.get("mrr") or 0.0) - 0.005:
            continue
        if float(row.get("ndcg_at_5") or 0.0) < float(baseline.get("ndcg_at_5") or 0.0) - 0.005:
            continue
        safe.append((name, row))
    best = max(
        safe,
        key=lambda item: (
            float(item[1].get("recall_at_5") or 0.0),
            float(item[1].get("mrr") or 0.0),
            float(item[1].get("ndcg_at_5") or 0.0),
            int(item[1].get("rescued_fts_failures") or 0),
            -float(item[1].get("rescue_fraction") or 1.0),
            -float(item[1].get("mean_latency_ms") or 1e9),
        ),
        default=None,
    )
    return {
        "best_safe_strategy": best[0] if best else None,
        "best_safe_metrics": best[1] if best else None,
        "safe_strategy_count": len(safe),
        "automatic_promotion_allowed": False,
        "reason": "human_review_required" if best else "no_strategy_meets_non_degradation_gate",
    }
