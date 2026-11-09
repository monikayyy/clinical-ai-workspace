"""Hybrid retrieval evaluation, privacy, and package-adapter regressions."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
from pydantic import ValidationError

from app.embedding_backends import (
    DeterministicHashEmbeddingBackend,
    FastEmbedBackend,
    LocalEmbeddingConfig,
    make_langchain_embeddings,
)
from app.hybrid_benchmark import build_hard_questions, evaluate_ranker, promotion_decision
from app.rerankers import FastEmbedReranker, KeywordOverlapTestReranker, LocalRerankerConfig
from app.retrieval_contracts import (
    ClinicalAccessScope,
    HybridRetrievalPolicy,
    HybridRetrievalRequest,
    RetrievalMode,
    RerankStrategy,
)
from app.retrieval_eval import FTS5Retriever, RetrievalDocument, load_synthetic_corpus
from app.retrieval_graph import (
    OptionalDependencyUnavailable,
    PatientScopedHybridWorkflow,
    RetrievalSecurityError,
)

ROOT = Path(__file__).resolve().parents[1]


def documents_for(patient='P'):
    return [
        RetrievalDocument('enc', patient, 'encounter_summary', 'Encounter', 'Primary care encounter summary. Active problem: hypertension.'),
        RetrievalDocument('med', patient, 'medication_reconciliation', 'Medication reconciliation', 'Medication reconciliation. Lisinopril 10 mg tablet.'),
        RetrievalDocument('obs', patient, 'observation_report', 'Observation', 'Laboratory or observation report. Hemoglobin A1c result: 7.2 %.'),
    ]


def request_for(documents, question='What are they taking?', mode=RetrievalMode.HYBRID_EVAL):
    return HybridRetrievalRequest(
        query_id='q',
        question=question,
        scope=ClinicalAccessScope(
            patient_key=documents[0].patient_key,
            allowed_document_ids=frozenset(document.id for document in documents),
        ),
        policy=HybridRetrievalPolicy(
            mode=mode,
            lexical_k=3,
            dense_k=3,
            fusion_k=3,
            rerank_k=3,
            final_k=3,
        ),
    )


def test_pydantic_contract_forbids_remote_egress_and_unknown_fields():
    with pytest.raises(ValidationError):
        HybridRetrievalPolicy(remote_embeddings_allowed=True)
    with pytest.raises(ValidationError):
        HybridRetrievalPolicy(remote_reranking_allowed=True)
    with pytest.raises(ValidationError):
        HybridRetrievalPolicy(remote_vector_db_allowed=True)
    with pytest.raises(ValidationError):
        HybridRetrievalPolicy(external_tracing_allowed=True)
    with pytest.raises(ValidationError):
        HybridRetrievalRequest(
            query_id='q', question='question', patient_id='model-supplied-patient',
            scope=ClinicalAccessScope(patient_key='P', allowed_document_ids=frozenset({'d'})),
        )


def test_policy_rejects_invalid_candidate_sizes():
    with pytest.raises(ValidationError):
        HybridRetrievalPolicy(fusion_k=20, rerank_k=10, final_k=11)
    with pytest.raises(ValidationError):
        HybridRetrievalPolicy(lexical_k=5, dense_k=5, fusion_k=10, rerank_k=5, final_k=5)


def test_graph_rejects_foreign_patient_before_embedding_or_reranking():
    allowed = documents_for('P-A')
    foreign = RetrievalDocument('foreign', 'P-B', 'medication_reconciliation', 'FOREIGN', 'SECRET FOREIGN SENTINEL')
    backend = DeterministicHashEmbeddingBackend()
    reranker = KeywordOverlapTestReranker()
    workflow = PatientScopedHybridWorkflow(backend, reranker)
    request = request_for(allowed)
    with pytest.raises(RetrievalSecurityError):
        workflow.run_reference(request, allowed + [foreign])
    assert backend.seen_texts == []
    assert reranker.seen_documents == []


def test_reference_hybrid_only_returns_authorized_document_ids():
    docs = documents_for()
    workflow = PatientScopedHybridWorkflow(DeterministicHashEmbeddingBackend(), KeywordOverlapTestReranker())
    response = workflow.run_reference(request_for(docs), docs)
    assert response.patient_leakage_count == 0
    assert response.remote_calls_allowed is False
    assert response.model_downloads_allowed is False
    assert response.external_tracing_allowed is False
    assert {item.document_id for item in response.candidates} <= {doc.id for doc in docs}
    assert response.semantic_available is True
    assert response.reranker_available is True


def test_external_tracing_causes_fail_closed(monkeypatch):
    docs = documents_for()
    monkeypatch.setenv('LANGSMITH_TRACING', 'true')
    workflow = PatientScopedHybridWorkflow(DeterministicHashEmbeddingBackend(), KeywordOverlapTestReranker())
    with pytest.raises(RetrievalSecurityError, match='tracing'):
        workflow.run_reference(request_for(docs), docs)


def test_fastembed_adapter_forces_local_files_only_and_specific_path(monkeypatch, tmp_path):
    captured = {}
    model_dir = tmp_path / 'embed-model'; model_dir.mkdir()

    class FakeTextEmbedding:
        def __init__(self, **kwargs): captured.update(kwargs)
        def query_embed(self, text): return iter([np.array([1.0, 0.0], dtype=np.float32)])
        def embed(self, texts): return iter(np.array([1.0, 0.0], dtype=np.float32) for _ in texts)

    module = types.ModuleType('fastembed')
    module.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, 'fastembed', module)
    backend = FastEmbedBackend(LocalEmbeddingConfig(
        model_name='BAAI/bge-small-en-v1.5', specific_model_path=str(model_dir)
    ))
    assert backend.available
    assert captured['local_files_only'] is True
    assert captured['specific_model_path'] == str(model_dir)
    assert backend.embed_query('query').tolist() == [1.0, 0.0]


def test_fastembed_adapter_refuses_missing_local_path(tmp_path):
    backend = FastEmbedBackend(LocalEmbeddingConfig(
        specific_model_path=str(tmp_path / 'missing-model')
    ))
    assert backend.available is False
    assert 'does not exist' in (backend.error or '')


def test_fastembed_reranker_forces_local_files_only_and_never_adds_candidates(monkeypatch, tmp_path):
    captured = {}
    model_dir = tmp_path / 'reranker-model'; model_dir.mkdir()

    class FakeTextCrossEncoder:
        def __init__(self, **kwargs): captured.update(kwargs)
        def rerank(self, query, documents): return iter([0.1, 0.9][:len(documents)])

    fastembed = types.ModuleType('fastembed')
    rerank_pkg = types.ModuleType('fastembed.rerank')
    cross = types.ModuleType('fastembed.rerank.cross_encoder')
    cross.TextCrossEncoder = FakeTextCrossEncoder
    monkeypatch.setitem(sys.modules, 'fastembed', fastembed)
    monkeypatch.setitem(sys.modules, 'fastembed.rerank', rerank_pkg)
    monkeypatch.setitem(sys.modules, 'fastembed.rerank.cross_encoder', cross)
    reranker = FastEmbedReranker(LocalRerankerConfig(specific_model_path=str(model_dir)))
    assert reranker.available
    assert captured['local_files_only'] is True
    assert captured['specific_model_path'] == str(model_dir)
    assert reranker.score('q', ['a','b']) == [0.1,0.9]


def test_langchain_adapter_is_optional_and_uses_same_local_backend(monkeypatch):
    backend = DeterministicHashEmbeddingBackend(dimensions=8)
    # Build a tiny compatible fake langchain-core module when the optional real
    # dependency is unavailable so adapter semantics are still exercised.
    if 'langchain_core.embeddings' not in sys.modules:
        pkg = types.ModuleType('langchain_core')
        embeddings = types.ModuleType('langchain_core.embeddings')
        class Embeddings: pass
        embeddings.Embeddings = Embeddings
        monkeypatch.setitem(sys.modules, 'langchain_core', pkg)
        monkeypatch.setitem(sys.modules, 'langchain_core.embeddings', embeddings)
    adapter = make_langchain_embeddings(backend)
    docs = adapter.embed_documents(['one','two'])
    query = adapter.embed_query('one')
    assert len(docs) == 2
    assert len(query) == 8


def test_hard_benchmark_exposes_real_lexical_gap_without_leakage():
    documents, base = load_synthetic_corpus(ROOT)
    hard = build_hard_questions(base)
    summary = evaluate_ranker(FTS5Retriever(), documents, hard)
    assert len(hard) == 1100
    assert summary.supported_questions == 900
    assert summary.patient_leakage_count == 0
    # Deterministic clinical alias expansion should close most of the cheap
    # lexical gap before semantic retrieval is considered.
    assert 0.95 <= summary.recall_at_5 <= 1.0
    assert summary.mrr >= 0.90
    assert summary.by_category['encounter_summary']['questions'] == 300
    assert summary.by_category['medication_reconciliation']['questions'] == 300
    assert summary.by_category['observation_report']['questions'] == 300


def test_promotion_gate_never_auto_enables_candidate():
    baseline = {'recall_at_5':0.90,'mrr':0.80,'ndcg_at_5':0.82,'patient_leakage_count':0}
    candidate = {'recall_at_5':0.96,'mrr':0.81,'ndcg_at_5':0.83,'patient_leakage_count':0}
    decision = promotion_decision(baseline,candidate)
    assert decision['metric_gate_passed'] is True
    assert decision['rank_quality_gate_passed'] is True
    assert decision['promote'] is False
    assert decision['automatic_promotion_allowed'] is False
    assert 'human_approval' in decision['reason']


def test_promotion_gate_rejects_recall_gain_when_ranking_quality_collapses():
    baseline = {'recall_at_5':0.6778,'mrr':0.5670,'ndcg_at_5':0.5952,'patient_leakage_count':0}
    candidate = {'recall_at_5':0.9489,'mrr':0.4116,'ndcg_at_5':0.5450,'patient_leakage_count':0}
    decision = promotion_decision(baseline,candidate)
    assert decision['recall_gate_passed'] is True
    assert decision['rank_quality_gate_passed'] is False
    assert decision['metric_gate_passed'] is False
    assert decision['promote'] is False


def test_promotion_gate_rejects_any_leakage_even_with_better_metrics():
    baseline = {'recall_at_5':0.68,'mrr':0.57,'ndcg_at_5':0.60,'patient_leakage_count':0}
    candidate = {'recall_at_5':1.0,'mrr':1.0,'ndcg_at_5':1.0,'patient_leakage_count':1}
    decision = promotion_decision(baseline,candidate)
    assert decision['metric_gate_passed'] is False
    assert decision['patient_leakage_ok'] is False


def test_hybrid_evaluation_cli_writes_hard_baseline(tmp_path):
    output = tmp_path / 'hybrid.json'
    result = subprocess.run(
        [sys.executable, str(ROOT/'scripts/evaluate_retrieval.py'), '--json-output', str(output)],
        cwd=ROOT, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert 'HYBRID_RETRIEVAL_EVALUATION_PASS' in result.stdout
    report = json.loads(output.read_text())
    assert report['benchmark_revision'] == 'hard-retrieval-v1'
    assert report['questions'] == 1100
    assert report['supported_questions'] == 900
    assert report['fts5']['patient_leakage_count'] == 0
    assert report['fts5']['recall_at_5'] >= 0.95
    assert report['privacy']['local_files_only'] is True
    assert report['promotion']['automatic_promotion_allowed'] is False


class CountingEmbeddingBackend(DeterministicHashEmbeddingBackend):
    def __init__(self):
        super().__init__(dimensions=16)
        self.document_batches = 0

    def embed_documents(self, texts):
        self.document_batches += 1
        return super().embed_documents(texts)


class ReverseReranker:
    name = 'reverse_test_reranker'
    available = True
    error = None

    def score(self, query, documents):
        return [float(index) for index, _ in enumerate(documents, 1)]


def test_document_embeddings_are_cached_per_patient_and_content():
    docs = documents_for()
    backend = CountingEmbeddingBackend()
    workflow = PatientScopedHybridWorkflow(backend, KeywordOverlapTestReranker())
    workflow.run_reference(request_for(docs, 'What are they taking?'), docs)
    workflow.run_reference(request_for(docs, 'Which therapy shows up in the med rec?'), docs)
    assert backend.document_batches == 1
    # Patient identity is part of the cache key even when content is identical.
    other = [RetrievalDocument(doc.id, 'OTHER', doc.source_kind, doc.title, doc.text) for doc in docs]
    workflow.run_reference(request_for(other, 'What are they taking?'), other)
    assert backend.document_batches == 2


def test_rank_blend_preserves_strong_lexical_anchor_against_bad_reranker():
    docs = documents_for()
    workflow = PatientScopedHybridWorkflow(DeterministicHashEmbeddingBackend(), ReverseReranker())
    request = HybridRetrievalRequest(
        query_id='blend', question='What are they taking?',
        scope=ClinicalAccessScope(patient_key='P', allowed_document_ids=frozenset(doc.id for doc in docs)),
        policy=HybridRetrievalPolicy(
            mode=RetrievalMode.HYBRID_EVAL, lexical_k=3, dense_k=3, fusion_k=3, rerank_k=3, final_k=3,
            use_reranker=True, rerank_strategy=RerankStrategy.RANK_BLEND,
        ),
    )
    response = workflow.run_reference(request, docs)
    assert response.backend.endswith('rank_blend')
    assert response.candidates[0].document_id == 'med'
    assert response.candidates[0].lexical_rank == 1


def test_replace_strategy_can_be_measured_as_legacy_comparator():
    docs = documents_for()
    workflow = PatientScopedHybridWorkflow(DeterministicHashEmbeddingBackend(), ReverseReranker())
    request = HybridRetrievalRequest(
        query_id='replace', question='What are they taking?',
        scope=ClinicalAccessScope(patient_key='P', allowed_document_ids=frozenset(doc.id for doc in docs)),
        policy=HybridRetrievalPolicy(
            mode=RetrievalMode.HYBRID_EVAL, lexical_k=3, dense_k=3, fusion_k=3, rerank_k=3, final_k=3,
            use_reranker=True, rerank_strategy=RerankStrategy.REPLACE,
        ),
    )
    response = workflow.run_reference(request, docs)
    assert response.backend.endswith('cross_encoder_replace')
    assert response.candidates[0].document_id != 'med'


def test_langgraph_real_adapter_if_installed():
    pytest.importorskip('langgraph')
    docs = documents_for()
    workflow = PatientScopedHybridWorkflow(DeterministicHashEmbeddingBackend(), KeywordOverlapTestReranker())
    response = workflow.run_langgraph(request_for(docs), docs)
    assert response.patient_leakage_count == 0
    assert response.candidates


