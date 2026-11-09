# Evaluation

## Benchmark

The public benchmark uses three compact synthetic files:

- `synthetic-data/documents.jsonl` — 445 synthetic documents;
- `synthetic-data/notes.jsonl` — 300 note distractors;
- `synthetic-data/questions.jsonl` — 400 base questions.

The hard-question builder deterministically expands these into 1,100 queries. Of those, 900 are answerable and are used for ranking metrics.

No real patient data is included.

## Metrics

The baseline tracks:

- Recall@1 / @3 / @5;
- Mean Reciprocal Rank;
- nDCG@5;
- mean/total query latency;
- source-kind misses;
- cross-patient leakage.

Reference results are stored in `benchmarks/fts5-baseline.json`.

| Metric | Result |
|---|---:|
| Recall@1 | 0.9556 |
| Recall@3 | 0.9667 |
| Recall@5 | 0.9778 |
| MRR | 0.9639 |
| nDCG@5 | 0.9674 |
| Leakage | 0 |

Category-level analysis shows perfect Recall@5 for encounter summaries and medication reconciliation in this fixture; the 20 top-five misses occur in observation/report questions.

## Reproduce

```bash
make benchmark
```

or:

```bash
python scripts/evaluate_retrieval.py --json-output runs/fts5-baseline.json
```

The lexical benchmark requires no model downloads.

## Experimental retrieval

The optional ML stack evaluates dense embeddings, RRF, local cross-encoder reranking, and decision/rescue logic. These components are isolated from the live default path.

```bash
pip install -r requirements-ml.txt
python scripts/evaluate_retrieval.py --hybrid --json-output runs/hybrid.json
```

FastEmbed and reranker adapters are configured for local model files/caches. External tracing is disabled by the benchmark process.

## Promotion rule

A candidate is not promoted because one metric improved. The current gate requires:

1. zero cross-patient leakage;
2. a meaningful Recall@5 improvement;
3. no material MRR regression;
4. no material nDCG@5 regression;
5. explicit human approval even after the metric gate passes.

This is intentionally conservative because a semantic retriever can increase candidate recall while making the correct evidence consistently rank lower.

## Failure analysis

The original lexical benchmark was too easy, so synonym, abbreviation, indirect, and colloquial variants were added to create measurable headroom.

The harder benchmark exposed a concentrated weakness in observation/report retrieval. That result motivated semantic and reranking experiments, but those paths remain evaluation-only until they improve ranking quality without weakening isolation.

The practical lesson is that an applied-ML evaluation should be able to reject an experiment. A benchmark that only confirms the desired architecture is not useful for model selection.
