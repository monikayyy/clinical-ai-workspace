"""Harder retrieval benchmark for lexical vs local semantic/hybrid retrieval.

The existing corpus intentionally validates the baseline and is now saturated by
FTS5. This module derives low-overlap paraphrases, abbreviations, and colloquial
queries from the same synthetic gold labels so semantic retrieval must earn its
complexity on a harder subset.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re
import time
from typing import Any, Protocol

from .retrieval_eval import RetrievalDocument, RetrievalHit, RetrievalQuestion

HARD_BENCHMARK_REVISION = "hard-retrieval-v1"


@dataclass(frozen=True)
class HardRetrievalQuestion:
    id: str
    patient_key: str
    question: str
    answerability: str
    expected_source_kind: str | None
    expected_fact: str | None
    category: str
    variant: str


@dataclass(frozen=True)
class DetailedRetrievalSummary:
    backend: str
    supported_questions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    patient_leakage_count: int
    source_kind_misses: int
    latency_ms_total: float
    latency_ms_mean: float
    by_category: dict[str, dict[str, float | int]]

    def public(self) -> dict[str, Any]:
        return asdict(self)


class Ranker(Protocol):
    name: str
    def rank(self, question: str, documents: list[RetrievalDocument], limit: int = 10) -> list[RetrievalHit]: ...


_MEASUREMENT_VARIANTS = {
    "systolic blood pressure": [
        "What's their BP reading?",
        "How high was the top blood-pressure number?",
        "What did the cuff show for the upper pressure?",
    ],
    "hemoglobin a1c": [
        "How was long-term sugar control measuring?",
        "What's the glycated hemoglobin result?",
        "What was the three-month glucose marker?",
    ],
    "peak expiratory flow": [
        "How strong was the breathing flow test?",
        "What was the peak blow reading?",
        "What's the airflow peak?",
    ],
    "thyrotropin [units/volume] in serum or plasma": [
        "What's the TSH result?",
        "How did the thyroid-stimulating hormone test come back?",
        "What was the thyroid control hormone level?",
    ],
    "headache days in prior 30 days": [
        "How often were migraines happening recently?",
        "How many headache days did they report this month?",
        "What's the monthly headache burden?",
    ],
    "pain severity - 0-10 verbal numeric rating": [
        "How bad was the pain?",
        "What was their pain score?",
        "How severe did they rate the pain?",
    ],
    "reflux symptom days in prior 14 days": [
        "How often was heartburn bothering them lately?",
        "How many recent days had acid reflux symptoms?",
        "What's the recent heartburn frequency?",
    ],
    "low density lipoprotein cholesterol": [
        "What was the bad cholesterol result?",
        "What's the LDL?",
        "What did the atherogenic cholesterol measure?",
    ],
    "nasal symptom days in prior 14 days": [
        "How often were nose or allergy symptoms happening lately?",
        "How many recent days had nasal symptoms?",
        "What's the recent nose-symptom frequency?",
    ],
    "body mass index": [
        "What's their BMI?",
        "What was the weight-for-height index?",
        "What body-size index was recorded?",
    ],
}


def _measurement_name(question: str) -> str:
    match = re.search(r"(?i)^what value is documented for\s+(.+?)\?$", question.strip())
    return (match.group(1) if match else question).casefold()


def hard_variants(question: RetrievalQuestion) -> list[tuple[str, str]]:
    if question.answerability != "supported" or not question.expected_source_kind:
        if question.answerability != "supported":
            return [
                ("negative_synonym", "Do we know their ABO or Rh blood group?"),
                ("negative_indirect", "Is a confirmed blood group recorded anywhere?"),
            ]
        return []
    if question.expected_source_kind == "encounter_summary":
        return [
            ("semantic", "What chronic health issue is on this person's chart?"),
            ("indirect", "What ongoing illness is being followed?"),
            ("colloquial", "What health problem are they dealing with?"),
        ]
    if question.expected_source_kind == "medication_reconciliation":
        return [
            ("semantic", "What are they taking?"),
            ("abbreviation", "Which therapy shows up in their med rec?"),
            ("indirect", "What drug therapy is recorded for them?"),
        ]
    if question.expected_source_kind == "observation_report":
        name = _measurement_name(question.question)
        variants = _MEASUREMENT_VARIANTS.get(name)
        if variants:
            return [("abbreviation_or_synonym", variants[0]), ("semantic", variants[1]), ("indirect", variants[2])]
        return [
            ("semantic", "What was the relevant clinical measurement?"),
            ("indirect", "What did that test show?"),
            ("colloquial", "What was the number?"),
        ]
    return []


def build_hard_questions(base_questions: list[RetrievalQuestion]) -> list[HardRetrievalQuestion]:
    output: list[HardRetrievalQuestion] = []
    for question in base_questions:
        category = question.expected_source_kind or "not_documented"
        for index, (variant, text) in enumerate(hard_variants(question), 1):
            output.append(HardRetrievalQuestion(
                id=f"{question.id}-hard-{index}",
                patient_key=question.patient_key,
                question=text,
                answerability=question.answerability,
                expected_source_kind=question.expected_source_kind,
                expected_fact=question.expected_fact,
                category=category,
                variant=variant,
            ))
    return output


def _dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances[:k]))


def evaluate_ranker(ranker: Ranker, documents: list[RetrievalDocument], questions: list[HardRetrievalQuestion]) -> DetailedRetrievalSummary:
    by_patient: dict[str, list[RetrievalDocument]] = {}
    for document in documents:
        by_patient.setdefault(document.patient_key, []).append(document)
    supported = [q for q in questions if q.answerability == "supported" and q.expected_source_kind]
    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal_sum = 0.0
    ndcg_sum = 0.0
    leakage = 0
    misses = 0
    total_latency = 0.0
    by_category_raw: dict[str, dict[str, float | int]] = {}
    for question in supported:
        scoped = by_patient.get(question.patient_key, [])
        started = time.perf_counter()
        ranking = ranker.rank(question.question, scoped, limit=10)
        total_latency += (time.perf_counter() - started) * 1000.0
        allowed = {doc.id for doc in scoped}
        if any(hit.document_id not in allowed for hit in ranking):
            leakage += 1
        target_rank = None
        relevances: list[int] = []
        for index, hit in enumerate(ranking, 1):
            relevant = int(hit.source_kind == question.expected_source_kind)
            relevances.append(relevant)
            if relevant and target_rank is None:
                target_rank = index
        stats = by_category_raw.setdefault(question.category, {"questions": 0, "hits_at_5": 0, "reciprocal_sum": 0.0})
        stats["questions"] = int(stats["questions"]) + 1
        if target_rank is None:
            misses += 1
        else:
            reciprocal_sum += 1.0 / target_rank
            stats["reciprocal_sum"] = float(stats["reciprocal_sum"]) + 1.0 / target_rank
            for cutoff in hits_at:
                if target_rank <= cutoff:
                    hits_at[cutoff] += 1
            if target_rank <= 5:
                stats["hits_at_5"] = int(stats["hits_at_5"]) + 1
        ideal_relevant = min(sum(1 for doc in scoped if doc.source_kind == question.expected_source_kind), 5)
        ideal = _dcg([1] * ideal_relevant, 5)
        ndcg_sum += (_dcg(relevances, 5) / ideal) if ideal else 0.0
    denom = max(1, len(supported))
    by_category: dict[str, dict[str, float | int]] = {}
    for category, stats in by_category_raw.items():
        count = max(1, int(stats["questions"]))
        by_category[category] = {
            "questions": int(stats["questions"]),
            "recall_at_5": round(int(stats["hits_at_5"]) / count, 4),
            "mrr": round(float(stats["reciprocal_sum"]) / count, 4),
        }
    return DetailedRetrievalSummary(
        backend=ranker.name,
        supported_questions=len(supported),
        recall_at_1=round(hits_at[1] / denom, 4),
        recall_at_3=round(hits_at[3] / denom, 4),
        recall_at_5=round(hits_at[5] / denom, 4),
        mrr=round(reciprocal_sum / denom, 4),
        ndcg_at_5=round(ndcg_sum / denom, 4),
        patient_leakage_count=leakage,
        source_kind_misses=misses,
        latency_ms_total=round(total_latency, 3),
        latency_ms_mean=round(total_latency / denom, 3),
        by_category=by_category,
    )


def promotion_decision(baseline: dict[str, Any], candidate: dict[str, Any] | None) -> dict[str, Any]:
    """Conservative human-gated promotion policy.

    Recall gains alone are insufficient: a candidate that retrieves the right
    source somewhere in the top five but consistently pushes it down the list is
    not acceptable for the live evidence path. Promotion eligibility therefore
    requires zero leakage plus non-degrading top-rank quality.
    """
    if not candidate:
        return {
            "promote": False,
            "automatic_promotion_allowed": False,
            "reason": "semantic_candidate_not_available",
        }
    leakage_ok = int(candidate.get("patient_leakage_count") or 0) == 0
    recall_delta = float(candidate.get("recall_at_5") or 0.0) - float(baseline.get("recall_at_5") or 0.0)
    mrr_delta = float(candidate.get("mrr") or 0.0) - float(baseline.get("mrr") or 0.0)
    ndcg_delta = float(candidate.get("ndcg_at_5") or 0.0) - float(baseline.get("ndcg_at_5") or 0.0)
    recall_gate = recall_delta >= 0.005
    rank_gate = mrr_delta >= -0.005 and ndcg_delta >= -0.005
    meaningful = recall_gate and rank_gate
    return {
        "promote": False,
        "automatic_promotion_allowed": False,
        "metric_gate_passed": bool(leakage_ok and meaningful),
        "patient_leakage_ok": leakage_ok,
        "recall_gate_passed": recall_gate,
        "rank_quality_gate_passed": rank_gate,
        "recall_at_5_delta": round(recall_delta, 4),
        "mrr_delta": round(mrr_delta, 4),
        "ndcg_at_5_delta": round(ndcg_delta, 4),
        "reason": (
            "candidate_beats_retrieval_gate_but_requires_human_approval"
            if leakage_ok and meaningful
            else "candidate_does_not_justify_added_complexity"
        ),
    }

