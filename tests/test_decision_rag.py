from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from app.decision_benchmark import evaluate_decision_strategy, stratified_questions
from app.decision_models import (
    ArcDecideBackend,
    D1DecisionBackend,
    HeuristicDecisionBackend,
    LocalDecisionModelConfig,
)
from app.decision_rag import (
    DecisionRetrievalPolicy,
    DecisionRetrievalSecurityError,
    PatientDecisionRetriever,
)
from app.hybrid_benchmark import build_hard_questions
from app.retrieval_eval import RetrievalDocument, load_synthetic_corpus


class SmartDecisionBackend:
    name = "smart_test_decider"
    available = True
    error = None

    def choose(self, state_text, options, *, instruction):
        q = state_text.casefold()
        scores = {}
        for key, description in options.items():
            d = description.casefold()
            score = 0.01
            if any(term in q for term in ("medication", "medicine", "taking", "therapy")) and any(term in d for term in ("medication", "drug", "therapy")):
                score += 10
            if any(term in q for term in ("blood", "result", "measurement", "tsh", "ldl", "a1c")) and any(term in d for term in ("observation", "measurement", "result", "laboratory")):
                score += 10
            if any(term in q for term in ("condition", "illness", "diagnosis", "problem")) and any(term in d for term in ("encounter", "condition", "diagnosis")):
                score += 10
            scores[key] = score
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}


def _docs(patient="P1"):
    return [
        RetrievalDocument("enc", patient, "encounter_summary", "enc.txt", "Document date: 2026-01-01\nActive condition: hypertension."),
        RetrievalDocument("obs", patient, "observation_report", "obs.txt", "Document date: 2026-01-02\nObservation: Systolic blood pressure\nResult: 128 mmHg."),
        RetrievalDocument("med", patient, "medication_reconciliation", "med.txt", "Document date: 2026-01-03\nMedication reconciliation\nLisinopril 10 mg daily."),
        RetrievalDocument("note", patient, "note", "note.txt", "Document date: 2026-01-04\nGeneral follow-up note."),
    ]


def test_navigation_routes_to_medication_source():
    retriever = PatientDecisionRetriever(SmartDecisionBackend())
    result = retriever.navigate("What medication are they taking?", _docs(), DecisionRetrievalPolicy(dense_rescue=False))
    assert result.hits
    assert result.hits[0].source_kind == "medication_reconciliation"
    assert result.route_probabilities["medication_reconciliation"] == max(result.route_probabilities.values())
    assert result.patient_leakage_count == 0


def test_navigation_routes_to_observation_source():
    retriever = PatientDecisionRetriever(SmartDecisionBackend())
    result = retriever.navigate("What was the blood pressure result?", _docs(), DecisionRetrievalPolicy(dense_rescue=False))
    assert result.hits[0].source_kind == "observation_report"


def test_mixed_patient_input_fails_before_decision():
    docs = _docs("P1") + [RetrievalDocument("foreign", "P2", "observation_report", "f.txt", "blood pressure result 999")]
    with pytest.raises(DecisionRetrievalSecurityError):
        PatientDecisionRetriever(SmartDecisionBackend()).navigate("blood pressure?", docs)


def test_joint_rerank_never_returns_out_of_scope_document():
    result = PatientDecisionRetriever(SmartDecisionBackend()).rerank_union(
        "What medication is documented?", _docs(), DecisionRetrievalPolicy(dense_rescue=False)
    )
    assert {hit.document_id for hit in result.hits} <= {doc.id for doc in _docs()}


def test_arc_backend_requires_local_model_directory(tmp_path):
    backend = ArcDecideBackend(LocalDecisionModelConfig(model_path=str(tmp_path / "missing"), backend="arc"))
    assert not backend.available
    assert "does not exist" in (backend.error or "")


def test_d1_requires_explicit_local_model_code_permission(tmp_path):
    model = tmp_path / "d1"; model.mkdir()
    backend = D1DecisionBackend(LocalDecisionModelConfig(model_path=str(model), backend="d1"))
    assert not backend.available
    assert "disabled" in (backend.error or "")


def test_arc_fails_closed_when_external_tracing_enabled(tmp_path, monkeypatch):
    model = tmp_path / "arc"; model.mkdir(); (model / "arc_decide.py").write_text("raise AssertionError('must not import')\n")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    backend = ArcDecideBackend(LocalDecisionModelConfig(model_path=str(model), backend="arc"))
    assert not backend.available
    assert "tracing" in (backend.error or "")


def test_heuristic_decider_probabilities_are_normalized():
    probs = HeuristicDecisionBackend().choose(
        "blood pressure result",
        {"a": "blood pressure observation", "b": "medication list"},
        instruction="pick",
    )
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["a"] > probs["b"]


def test_stratified_sample_contains_all_supported_categories():
    root = Path(__file__).resolve().parents[1]
    _, base = load_synthetic_corpus(root)
    hard = build_hard_questions(base)
    sample = stratified_questions(hard, 30)
    categories = Counter(q.category for q in sample)
    assert set(categories) >= {"encounter_summary", "medication_reconciliation", "observation_report"}
    assert max(categories.values()) - min(categories.values()) <= 1


def test_dependency_free_decision_benchmark_has_zero_leakage():
    root = Path(__file__).resolve().parents[1]
    documents, base = load_synthetic_corpus(root)
    sample = stratified_questions(build_hard_questions(base), 15)
    summary = evaluate_decision_strategy(
        PatientDecisionRetriever(HeuristicDecisionBackend()),
        documents,
        sample,
        "decision_navigation",
        DecisionRetrievalPolicy(dense_rescue=False),
    )
    assert summary.supported_questions == 15
    assert summary.patient_leakage_count == 0
    assert 0.0 <= summary.recall_at_5 <= 1.0

def test_arc_local_client_can_be_loaded_without_network(tmp_path, monkeypatch):
    model = tmp_path / "arc"; model.mkdir()
    (model / "arc_decide.py").write_text(
        "class Decider:\n"
        "    def __init__(self, path, device='cpu', dtype='float32'):\n"
        "        self.path=path\n"
        "    def decide(self, state, questions):\n"
        "        opts=questions['choice']['criteria']\n"
        "        keys=list(opts)\n"
        "        return {'choice': {'type':'choice','choice':keys[0],'probabilities':{k:(1.0 if i==0 else 0.0) for i,k in enumerate(keys)}}}\n"
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    backend = ArcDecideBackend(LocalDecisionModelConfig(model_path=str(model), backend="arc", device="cpu", dtype="float32"))
    assert backend.available, backend.error
    probs = backend.choose("synthetic", {"a":"first", "b":"second"}, instruction="pick")
    assert probs == {"a": 1.0, "b": 0.0}

def _write_fake_arc_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "arc_decide.py").write_text(
        "class Decider:\n"
        "    def __init__(self, path, device='cpu', dtype='float32'):\n"
        "        self.path=path\n"
        "    def decide(self, state, questions):\n"
        "        opts=questions['choice']['criteria']\n"
        "        keys=list(opts)\n"
        "        # Deterministic distribution with the first option highest.\n"
        "        raw={k: float(len(keys)-i) for i,k in enumerate(keys)}\n"
        "        total=sum(raw.values())\n"
        "        return {'choice': {'type':'choice','choice':keys[0],'probabilities':{k:v/total for k,v in raw.items()}}}\n"
    )


