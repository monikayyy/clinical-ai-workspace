#!/usr/bin/env python3
"""Evaluate FTS5 against local FastEmbed + optional cross-encoder reranking.

This script uses the shipped synthetic corpus only. It never opens workspace.vault.
FastEmbed is configured with `local_files_only=True`; models must already be in a
local cache or supplied with an explicit local model path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embedding_backends import FastEmbedBackend, LocalEmbeddingConfig
from app.hybrid_benchmark import build_hard_questions, evaluate_ranker, promotion_decision
from app.rerankers import FastEmbedReranker, LocalRerankerConfig
from app.retrieval_contracts import ClinicalAccessScope, HYBRID_RETRIEVAL_REVISION, HybridRetrievalPolicy, HybridRetrievalRequest, RetrievalMode, RerankStrategy
from app.retrieval_eval import FTS5Retriever, RetrievalHit, load_synthetic_corpus
from app.retrieval_graph import OptionalDependencyUnavailable, PatientScopedHybridWorkflow


class HybridRanker:
    name = "langgraph_fastembed_hybrid"

    def __init__(self, workflow: PatientScopedHybridWorkflow, *, use_langgraph: bool, strategy: RerankStrategy):
        self.workflow = workflow
        self.use_langgraph = use_langgraph
        self.strategy = strategy
        self.name = f"langgraph_fastembed_{strategy.value}"
        self.total_queries = 0
        self.semantic_queries = 0

    def rank(self, question, documents, limit=10):
        scope = ClinicalAccessScope(
            patient_key=documents[0].patient_key,
            allowed_document_ids=frozenset(document.id for document in documents),
        )
        use_reranker = self.strategy != RerankStrategy.NONE
        policy = HybridRetrievalPolicy(
            mode=RetrievalMode.HYBRID_EVAL,
            lexical_k=min(40, max(10, len(documents))),
            dense_k=min(40, max(10, len(documents))),
            fusion_k=min(20, max(5, len(documents))),
            rerank_k=min(20, max(5, len(documents))),
            final_k=min(limit, 5),
            use_reranker=use_reranker,
            rerank_strategy=self.strategy,
        )
        request = HybridRetrievalRequest(query_id="benchmark", question=question, scope=scope, policy=policy)
        if self.use_langgraph:
            response = self.workflow.run_langgraph(request, documents)
        else:
            response = self.workflow.run_reference(request, documents)
        self.total_queries += 1
        self.semantic_queries += int(response.semantic_used)
        return [
            RetrievalHit(item.document_id, item.source_kind, float(item.blended_score or item.rerank_score or item.fusion_score or 0.0), response.backend)
            for item in response.candidates[:limit]
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hybrid', action='store_true', help='Evaluate local FastEmbed hybrid retrieval.')
    parser.add_argument('--reference-engine', action='store_true', help='Use dependency-free workflow executor instead of LangGraph. Intended for tests/debugging.')
    parser.add_argument('--embedding-model-name', default='BAAI/bge-small-en-v1.5')
    parser.add_argument('--embedding-cache', type=Path)
    parser.add_argument('--embedding-path', type=Path)
    parser.add_argument('--reranker-model-name', default='Xenova/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--reranker-cache', type=Path)
    parser.add_argument('--reranker-path', type=Path)
    parser.add_argument('--no-rerank', action='store_true', help='Alias for --strategy none.')
    parser.add_argument('--strategy', choices=[item.value for item in RerankStrategy], default=RerankStrategy.RANK_BLEND.value, help='How cross-encoder scores affect final ordering.')
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()

    # Disable external tracing and Hugging Face network fallback for this process.
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    documents, base_questions = load_synthetic_corpus(ROOT)
    hard_questions = build_hard_questions(base_questions)
    baseline = evaluate_ranker(FTS5Retriever(), documents, hard_questions).public()
    report = {
        'revision': HYBRID_RETRIEVAL_REVISION,
        'benchmark_revision': 'hard-retrieval-v1',
        'documents': len(documents),
        'questions': len(hard_questions),
        'supported_questions': sum(q.answerability == 'supported' for q in hard_questions),
        'fts5': baseline,
        'hybrid': None,
        'privacy': {
            'patient_scope_before_embedding': True,
            'local_files_only': True,
            'remote_embeddings_allowed': False,
            'remote_reranking_allowed': False,
            'remote_vector_db_allowed': False,
            'external_tracing_allowed': False,
            'persistent_vector_store': False,
        },
    }

    if args.hybrid:
        embedding = FastEmbedBackend(LocalEmbeddingConfig(
            model_name=args.embedding_model_name,
            cache_dir=str(args.embedding_cache) if args.embedding_cache else None,
            specific_model_path=str(args.embedding_path) if args.embedding_path else None,
        ))
        strategy = RerankStrategy.NONE if args.no_rerank else RerankStrategy(args.strategy)
        reranker = None if strategy == RerankStrategy.NONE else FastEmbedReranker(LocalRerankerConfig(
            model_name=args.reranker_model_name,
            cache_dir=str(args.reranker_cache) if args.reranker_cache else None,
            specific_model_path=str(args.reranker_path) if args.reranker_path else None,
        ))
        report['hybrid_runtime'] = {
            'embedding_available': embedding.available,
            'embedding_error': embedding.error,
            'embedding_model': args.embedding_model_name,
            'reranker_available': bool(reranker and reranker.available),
            'reranker_error': reranker.error if reranker else None,
            'reranker_model': args.reranker_model_name if reranker else None,
            'engine': 'reference' if args.reference_engine else 'langgraph',
            'strategy': strategy.value,
        }
        if embedding.available:
            workflow = PatientScopedHybridWorkflow(embedding_backend=embedding, reranker=reranker)
            ranker = HybridRanker(workflow, use_langgraph=not args.reference_engine, strategy=strategy)
            try:
                hybrid = evaluate_ranker(ranker, documents, hard_questions).public()
                hybrid['semantic_query_fraction'] = round(
                    ranker.semantic_queries / max(1, ranker.total_queries), 4
                )
            except OptionalDependencyUnavailable as exc:
                report['hybrid_runtime']['graph_error'] = str(exc)
            else:
                report['hybrid'] = hybrid

    report['promotion'] = promotion_decision(report['fts5'], report.get('hybrid'))

    print('Hybrid retrieval evaluation revision:', report['revision'])
    print('Synthetic documents:', report['documents'])
    print('Hard questions:', report['questions'])
    print('Supported hard questions:', report['supported_questions'])
    print('FTS5 recall@1:', f"{baseline['recall_at_1']:.1%}")
    print('FTS5 recall@3:', f"{baseline['recall_at_3']:.1%}")
    print('FTS5 recall@5:', f"{baseline['recall_at_5']:.1%}")
    print('FTS5 MRR:', f"{baseline['mrr']:.4f}")
    print('FTS5 nDCG@5:', f"{baseline['ndcg_at_5']:.4f}")
    print('Cross-patient leakage:', baseline['patient_leakage_count'])
    if args.hybrid:
        runtime = report.get('hybrid_runtime') or {}
        print('Embedding available:', runtime.get('embedding_available'))
        print('Reranker available:', runtime.get('reranker_available'))
        if runtime.get('embedding_error'):
            print('Embedding note:', runtime['embedding_error'])
        if runtime.get('reranker_error'):
            print('Reranker note:', runtime['reranker_error'])
        if runtime.get('graph_error'):
            print('Graph note:', runtime['graph_error'])
        if report.get('hybrid'):
            hybrid = report['hybrid']
            print('Hybrid recall@5:', f"{hybrid['recall_at_5']:.1%}")
            print('Hybrid MRR:', f"{hybrid['mrr']:.4f}")
            print('Hybrid nDCG@5:', f"{hybrid['ndcg_at_5']:.4f}")
            print('Hybrid leakage:', hybrid['patient_leakage_count'])
            print('Hybrid semantic query fraction:', f"{hybrid.get('semantic_query_fraction', 1.0):.1%}")
    print('Promotion decision:', report['promotion']['reason'])
    print('Automatic promotion allowed:', report['promotion']['automatic_promotion_allowed'])

    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print('JSON report:', args.json_output)

    if baseline['patient_leakage_count'] != 0:
        raise SystemExit('HYBRID_RETRIEVAL_EVALUATION_FAIL: lexical baseline leaked across patients')

    # A baseline-only run is complete when FTS5 succeeds.  But when --hybrid is
    # explicitly requested, do not print PASS unless a real hybrid metric block
    # was produced.  This prevents missing optional packages/models from looking
    # like a successful semantic benchmark.
    if args.hybrid and not isinstance(report.get('hybrid'), dict):
        runtime = report.get('hybrid_runtime') or {}
        notes = []
        if runtime.get('embedding_error'):
            notes.append('embedding=' + str(runtime['embedding_error']))
        if runtime.get('reranker_error'):
            notes.append('reranker=' + str(runtime['reranker_error']))
        if runtime.get('graph_error'):
            notes.append('graph=' + str(runtime['graph_error']))
        print('HYBRID_RETRIEVAL_EVALUATION_INCOMPLETE')
        if notes:
            print('Reason:', '; '.join(notes))
        print('Install optional packages into this project environment with:')
        print('  uv pip install -r requirements-ml.txt')
        print('Verify that the configured local embedding/reranker model paths exist, then rerun.')
        raise SystemExit(2)

    print('HYBRID_RETRIEVAL_EVALUATION_PASS')


if __name__ == '__main__':
    main()
