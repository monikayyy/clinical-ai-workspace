from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.decision_rescue import (
    ConfidenceGatedDecisionRetriever,
    DecisionRescuePolicy,
    lexical_confidence,
    rescue_query_profile,
)
from app.decision_rescue_benchmark import evaluate_gate, evaluate_rescue_strategy
from app.decision_rag import DecisionRetrievalSecurityError
from app.embedding_backends import DeterministicHashEmbeddingBackend
from app.hybrid_benchmark import build_hard_questions
from app.retrieval_eval import FTS5Retriever, RetrievalDocument, RetrievalHit, load_synthetic_corpus

ROOT = Path(__file__).resolve().parents[1]


class RecordingDecisionBackend:
    name = "recording_decider"
    available = True
    error = None

    def __init__(self):
        self.calls = []

    def choose(self, state_text, options, *, instruction):
        self.calls.append((state_text, dict(options), instruction))
        q = state_text.casefold()
        scores = {}
        for key, value in options.items():
            text = value.casefold()
            score = 0.01
            if "taking" in q and "medication" in text:
                score += 10
            if any(term in q for term in ("blood pressure", "sugar control", "migraine", "result")) and "observation" in text:
                score += 10
            scores[key] = score
        total = sum(scores.values())
        return {key: value / total for key, value in scores.items()}


class FlatDecisionBackend(RecordingDecisionBackend):
    name = "flat_decider"

    def choose(self, state_text, options, *, instruction):
        self.calls.append((state_text, dict(options), instruction))
        return {key: 1.0 / len(options) for key in options}


def _docs(patient="P1"):
    return [
        RetrievalDocument("enc", patient, "encounter_summary", "enc.txt", "Active condition: hypertension."),
        RetrievalDocument("obs", patient, "observation_report", "obs.txt", "Observation report. Hemoglobin A1c result 7.2 %."),
        RetrievalDocument("med", patient, "medication_reconciliation", "med.txt", "Medication reconciliation. Lisinopril 10 mg daily."),
        RetrievalDocument("note", patient, "note", "note.txt", "General follow-up narrative."),
    ]


def test_rescue_profile_recognizes_low_overlap_observation_phrases():
    assert rescue_query_profile("How was long-term sugar control measuring?")["preferred_source_kinds"] == {"observation_report"}
    assert rescue_query_profile("What body-size index was recorded?")["preferred_source_kinds"] == {"observation_report"}
    assert rescue_query_profile("How often were nose or allergy symptoms happening lately?")["preferred_source_kinds"] == {"observation_report"}


def test_strong_fts_gate_never_calls_embedding_or_decision():
    embedding = DeterministicHashEmbeddingBackend()
    decision = RecordingDecisionBackend()
    retriever = ConfidenceGatedDecisionRetriever(decision, embedding)
    result = retriever.retrieve("What are they taking?", _docs(), DecisionRescuePolicy(use_dense=True, use_decision=True))
    assert result.confidence.strong is True
    assert result.semantic_used is False
    assert result.hits[0].source_kind == "medication_reconciliation"
    assert embedding.seen_texts == []
    assert decision.calls == []


def test_low_confidence_gate_runs_dense_and_arc_rescue(monkeypatch):
    embedding = DeterministicHashEmbeddingBackend()
    decision = RecordingDecisionBackend()
    retriever = ConfidenceGatedDecisionRetriever(decision, embedding)
    monkeypatch.setattr(
        retriever.fts,
        "rank",
        lambda question, documents, limit=10: [RetrievalHit("note", "note", 2.0, "forced")],
    )
    monkeypatch.setattr(
        retriever,
        "_dense",
        lambda question, documents: ([
            RetrievalHit("obs", "observation_report", 0.99, "dense"),
            RetrievalHit("note", "note", 0.20, "dense"),
        ], 3.0),
    )
    result = retriever.retrieve(
        "How was long-term sugar control measuring?",
        _docs(),
        DecisionRescuePolicy(use_dense=True, use_decision=True),
    )
    assert result.confidence.strong is False
    assert result.semantic_used is True
    assert result.dense_used is True
    assert result.decision_used is True
    assert result.decision_accepted is True
    assert result.hits[0].source_kind == "observation_report"
    assert decision.calls


def test_low_confidence_arc_probability_margin_can_fail_back_to_rrf(monkeypatch):
    embedding = DeterministicHashEmbeddingBackend()
    decision = FlatDecisionBackend()
    retriever = ConfidenceGatedDecisionRetriever(decision, embedding)
    monkeypatch.setattr(
        retriever.fts,
        "rank",
        lambda question, documents, limit=10: [RetrievalHit("note", "note", 2.0, "forced")],
    )
    monkeypatch.setattr(
        retriever,
        "_dense",
        lambda question, documents: ([
            RetrievalHit("obs", "observation_report", 0.99, "dense"),
            RetrievalHit("note", "note", 0.20, "dense"),
        ], 3.0),
    )
    result = retriever.retrieve(
        "How was long-term sugar control measuring?",
        _docs(),
        DecisionRescuePolicy(
            use_dense=True,
            use_decision=True,
            decision_min_top_probability=0.60,
            decision_min_margin=0.20,
        ),
    )
    assert result.decision_used is True
    assert result.decision_accepted is False
    assert result.backend.endswith("arc_fallback")
    assert result.hits


def test_mixed_patient_scope_fails_before_any_semantic_call():
    embedding = DeterministicHashEmbeddingBackend()
    decision = RecordingDecisionBackend()
    docs = _docs("P1") + [RetrievalDocument("foreign", "P2", "observation_report", "foreign", "Result 999")]
    with pytest.raises(DecisionRetrievalSecurityError):
        ConfidenceGatedDecisionRetriever(decision, embedding).retrieve(
            "What was the result?", docs, DecisionRescuePolicy(use_dense=True, use_decision=True)
        )
    assert embedding.seen_texts == []
    assert decision.calls == []


def test_full_900_gate_targets_all_current_fts_misses_without_touching_successes():
    documents, base = load_synthetic_corpus(ROOT)
    hard = build_hard_questions(base)
    summary = evaluate_gate(documents, hard, DecisionRescuePolicy(use_dense=False, use_decision=False))
    assert summary.supported_questions == 900
    assert summary.fts_failures == 20
    assert summary.rescue_queries == 20
    assert summary.fts_failures_triggered == 20
    assert summary.fts_failure_trigger_rate == 1.0
    assert summary.fts_successes_triggered == 0
    assert summary.patient_leakage_count == 0


def test_dependency_free_rescue_benchmark_preserves_fts_successes():
    documents, base = load_synthetic_corpus(ROOT)
    hard = build_hard_questions(base)
    decision = RecordingDecisionBackend()
    # Arc-only evaluation uses lexical candidates on the gated subset and should
    # leave every strong FTS query byte-for-byte in the lexical path.
    retriever = ConfidenceGatedDecisionRetriever(decision_backend=decision, embedding_backend=None)
    summary = evaluate_rescue_strategy(
        retriever,
        documents,
        hard,
        DecisionRescuePolicy(use_dense=False, use_decision=True),
        "arc_rescue_test",
    )
    assert summary.supported_questions == 900
    assert summary.rescue_queries == 20
    assert summary.damaged_fts_successes == 0
    assert summary.decision_queries == 20
    assert summary.patient_leakage_count == 0


def test_dense_document_embeddings_are_cached_per_patient_source_hash(monkeypatch):
    embedding = DeterministicHashEmbeddingBackend()
    retriever = ConfidenceGatedDecisionRetriever(decision_backend=None, embedding_backend=embedding)
    monkeypatch.setattr(
        retriever.fts,
        "rank",
        lambda question, documents, limit=10: [RetrievalHit("note", "note", 1.0, "forced")],
    )
    policy = DecisionRescuePolicy(use_dense=True, use_decision=False)
    retriever.retrieve("How was long-term sugar control measuring?", _docs(), policy)
    first_seen = list(embedding.seen_texts)
    retriever.retrieve("How often were migraines happening recently?", _docs(), policy)
    second_seen = embedding.seen_texts[len(first_seen):]
    # Second rescue embeds only the new query; document vectors are reused in RAM.
    assert len(first_seen) == 1 + len(_docs())
    assert len(second_seen) == 1


def test_rescue_winner_gate_rejects_any_strategy_that_damages_fts_successes():
    from app.decision_rescue_benchmark import choose_rescue_winner
    baseline = {"recall_at_5": 0.9778, "mrr": 0.9639, "ndcg_at_5": 0.9674}
    safe = {
        "recall_at_5": 0.99,
        "mrr": 0.965,
        "ndcg_at_5": 0.968,
        "patient_leakage_count": 0,
        "damaged_fts_successes": 0,
        "rescued_fts_failures": 10,
        "rescue_fraction": 0.02,
        "mean_latency_ms": 5.0,
    }
    damaged = dict(safe, damaged_fts_successes=1, recall_at_5=1.0)
    result = choose_rescue_winner(baseline, {"safe": safe, "damaged": damaged})
    assert result["best_safe_strategy"] == "safe"
    assert result["automatic_promotion_allowed"] is False
