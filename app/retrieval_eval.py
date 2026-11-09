"""Patient-scoped retrieval backends and synthetic benchmark utilities.

FTS5 is the production-safe baseline. Optional dense embeddings and cross-encoder
reranking are evaluation-only until they outperform the lexical baseline on the
shipped synthetic corpus. All candidate sets are patient-scoped *before* ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Protocol

from .fact_store import extract_facts
from .medication_event_workflow import EligibleSource

RETRIEVAL_REVISION = "retrieval-eval-v1"
DEFAULT_TOP_K = (1, 3, 5)


@dataclass(frozen=True)
class RetrievalDocument:
    id: str
    patient_key: str
    source_kind: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalQuestion:
    id: str
    patient_key: str
    question: str
    answerability: str
    expected_source_kind: str | None
    expected_fact: str | None


@dataclass(frozen=True)
class RetrievalHit:
    document_id: str
    source_kind: str
    score: float
    backend: str


@dataclass(frozen=True)
class BenchmarkSummary:
    backend: str
    supported_questions: int
    not_documented_questions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    patient_leakage_count: int
    source_kind_misses: int
    available: bool = True
    note: str | None = None

    def public(self) -> dict[str, Any]:
        return asdict(self)


class Retriever(Protocol):
    name: str
    def rank(self, question: str, documents: list[RetrievalDocument], limit: int = 10) -> list[RetrievalHit]: ...


def _tokens(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,40}", text.casefold())))[:32]


def query_profile(question: str) -> dict[str, Any]:
    """Deterministic clinical query normalization for the lexical baseline.

    Keep this deliberately small and auditable. It expands common user-facing
    synonyms/abbreviations into words likely to occur in the synthetic/reviewed
    record while also providing a source-kind prior. It never changes patient
    scope or invents clinical facts.
    """
    q = question.casefold()
    medication = re.search(
        r"\b(?:medication|medicine|meds?|drug|therapy|therapies|taking|med\s*rec|prescription)\b",
        q,
    )
    condition = re.search(
        r"\b(?:condition|diagnosis|problem|disease|illness|chronic\s+health|health\s+issue|ongoing\s+illness)\b",
        q,
    )
    measurement = re.search(
        r"\b(?:value|result|measurement|test|number|reading|bp|blood[- ]?pressure|cuff|a1c|glycated|glucose|"
        r"cholesterol|ldl|creatinine|egfr|flow|airflow|peak\s+blow|tsh|thyroid|headache|migraine|pain|"
        r"reflux|heartburn|nasal|bmi|weight[- ]?for[- ]?height|symptom)\b",
        q,
    )
    if medication:
        return {
            "kind": "medication",
            "preferred_source_kinds": {"medication_reconciliation"},
            "expansion": ["medication", "reconciliation", "therapy", "drug"],
        }
    if condition:
        return {
            "kind": "condition",
            "preferred_source_kinds": {"encounter_summary"},
            "expansion": ["condition", "problem", "active", "diagnosis"],
        }
    if measurement:
        expansion = ["observation", "result", "measurement"]
        alias_groups = [
            (r"\b(?:bp|blood[- ]?pressure|cuff|top\s+blood)\b", ["blood", "pressure", "systolic"]),
            (r"\b(?:a1c|glycated|three[- ]month\s+glucose|sugar\s+control)\b", ["hemoglobin", "a1c"]),
            (r"\b(?:tsh|thyroid[- ]stimulating|thyroid\s+control)\b", ["thyrotropin", "thyroid"]),
            (r"\b(?:ldl|bad\s+cholesterol|atherogenic\s+cholesterol)\b", ["low", "density", "lipoprotein", "cholesterol"]),
            (r"\b(?:bmi|weight[- ]?for[- ]?height|body[- ]size\s+index)\b", ["body", "mass", "index"]),
            (r"\b(?:peak\s+blow|airflow|breathing\s+flow)\b", ["peak", "expiratory", "flow"]),
            (r"\b(?:migraine|headache)\b", ["headache", "days"]),
            (r"\b(?:pain\s+score|pain\s+severity|how\s+bad\s+was\s+the\s+pain)\b", ["pain", "severity"]),
            (r"\b(?:heartburn|acid\s+reflux|reflux)\b", ["reflux", "symptom", "days"]),
            (r"\b(?:nasal|nose\s+symptom|allergy\s+symptom)\b", ["nasal", "symptom", "days"]),
        ]
        for pattern, aliases in alias_groups:
            if re.search(pattern, q):
                expansion.extend(aliases)
        return {
            "kind": "measurement",
            "preferred_source_kinds": {"observation_report"},
            "expansion": list(dict.fromkeys(expansion)),
        }
    return {"kind": "open", "preferred_source_kinds": set(), "expansion": []}


def _fts_query(question: str) -> str:
    stop = {"what", "which", "the", "is", "are", "in", "for", "patient", "reviewed", "synthetic", "source", "records", "documented", "appears"}
    profile = query_profile(question)
    tokens = [token for token in _tokens(question) if token not in stop]
    tokens.extend(profile["expansion"])
    tokens = list(dict.fromkeys(tokens))[:32]
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


class FTS5Retriever:
    name = "fts5"

    def rank(self, question: str, documents: list[RetrievalDocument], limit: int = 10) -> list[RetrievalHit]:
        # Security boundary: caller passes documents for one authorized patient
        # only. A per-request in-memory table prevents another patient's content
        # from affecting either results or BM25 corpus statistics.
        if not documents:
            return []
        query = _fts_query(question)
        if not query:
            return []
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("CREATE VIRTUAL TABLE docs USING fts5(document_id UNINDEXED, source_kind UNINDEXED, title, body)")
            conn.executemany(
                "INSERT INTO docs(document_id,source_kind,title,body) VALUES(?,?,?,?)",
                [(doc.id, doc.source_kind, doc.title, doc.text) for doc in documents],
            )
            rows = conn.execute(
                "SELECT document_id,source_kind,bm25(docs,0,0,4.0,1.0) AS rank "
                "FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT ?",
                (query, max(1, min(max(limit * 5, 20), 50))),
            ).fetchall()
            # SQLite bm25 assigns better matches more-negative/lower scores. Add
            # a deterministic source-kind prior from the request profile; this is
            # the intended request-specific retrieval behavior, not patient data.
            preferred = query_profile(question)["preferred_source_kinds"]
            scored = []
            for row in rows:
                score = -float(row[2]) + (10.0 if row[1] in preferred else 0.0)
                scored.append(RetrievalHit(row[0], row[1], score, self.name))
            scored.sort(key=lambda hit: hit.score, reverse=True)
            return scored[:limit]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()


class SentenceTransformerRetriever:
    """Optional local-only dense retriever.

    No model is downloaded. `local_files_only=True` and `trust_remote_code=False`
    are mandatory. Configure a locally cached model or local path with
    RETRIEVAL_EMBEDDING_MODEL.
    """
    name = "dense"

    def __init__(self, model_name_or_path: str | None = None):
        self.model_name_or_path = model_name_or_path or os.getenv("RETRIEVAL_EMBEDDING_MODEL", "")
        self._model = None
        self.error: str | None = None
        if not self.model_name_or_path:
            self.error = "RETRIEVAL_EMBEDDING_MODEL is not configured"
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name_or_path,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:  # optional evaluation dependency
            self.error = f"{type(exc).__name__}: local embedding model unavailable"

    @property
    def available(self) -> bool:
        return self._model is not None

    def rank(self, question: str, documents: list[RetrievalDocument], limit: int = 10) -> list[RetrievalHit]:
        if not self._model or not documents:
            return []
        texts = [doc.title + "\n" + doc.text for doc in documents]
        query_vector = self._model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        doc_vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        scored = []
        for doc, vector in zip(documents, doc_vectors):
            score = float(sum(float(a) * float(b) for a, b in zip(query_vector, vector)))
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [RetrievalHit(doc.id, doc.source_kind, score, self.name) for score, doc in scored[:limit]]


class CrossEncoderReranker:
    """Optional local-only reranker for an already patient-scoped candidate set."""
    name = "cross_encoder"

    def __init__(self, model_name_or_path: str | None = None):
        self.model_name_or_path = model_name_or_path or os.getenv("RETRIEVAL_RERANKER_MODEL", "")
        self._model = None
        self.error: str | None = None
        if not self.model_name_or_path:
            self.error = "RETRIEVAL_RERANKER_MODEL is not configured"
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name_or_path,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: local reranker unavailable"

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(self, question: str, documents: list[RetrievalDocument], hits: list[RetrievalHit], limit: int = 10) -> list[RetrievalHit]:
        if not self._model or not hits:
            return hits[:limit]
        by_id = {doc.id: doc for doc in documents}
        candidates = [by_id[hit.document_id] for hit in hits if hit.document_id in by_id]
        pairs = [(question, doc.title + "\n" + doc.text) for doc in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, candidates), key=lambda item: float(item[0]), reverse=True)
        return [RetrievalHit(doc.id, doc.source_kind, float(score), self.name) for score, doc in ranked[:limit]]


def reciprocal_rank_fusion(rankings: Iterable[list[RetrievalHit]], *, k: int = 60, limit: int = 10) -> list[RetrievalHit]:
    totals: dict[str, tuple[float, str]] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            score, kind = totals.get(hit.document_id, (0.0, hit.source_kind))
            totals[hit.document_id] = (score + 1.0 / (k + rank), kind)
    ordered = sorted(totals.items(), key=lambda item: item[1][0], reverse=True)
    return [RetrievalHit(doc_id, kind, score, "hybrid_rrf") for doc_id, (score, kind) in ordered[:limit]]


def load_synthetic_corpus(root: Path) -> tuple[list[RetrievalDocument], list[RetrievalQuestion]]:
    documents: list[RetrievalDocument] = []
    with (root / "synthetic-data" / "documents.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            documents.append(RetrievalDocument(
                id=str(row["document_id"]),
                patient_key=str(row["patient_key"]),
                source_kind=str(row["kind"]),
                title=str(row["filename"]),
                text=str(row["gold_verified_text"]),
            ))
    # Notes are useful adversarial distractors, but gold retrieval questions in
    # this synthetic corpus point to document source kinds.
    with (root / "synthetic-data" / "notes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            documents.append(RetrievalDocument(
                id=str(row["note_id"]), patient_key=str(row["patient_key"]),
                source_kind="note", title=str(row["title"]), text=str(row["body"]),
            ))
    questions: list[RetrievalQuestion] = []
    with (root / "synthetic-data" / "questions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            questions.append(RetrievalQuestion(
                id=str(row["question_id"]), patient_key=str(row["patient_key"]),
                question=str(row["question"]), answerability=str(row["answerability"]),
                expected_source_kind=row.get("expected_source_kind"), expected_fact=row.get("expected_fact"),
            ))
    return documents, questions




def expand_benchmark_questions(questions: list[RetrievalQuestion]) -> list[RetrievalQuestion]:
    """Add deterministic paraphrases without changing the gold source/fact.

    These variants exercise request-specific routing rather than merely matching
    the exact wording emitted by the synthetic-data generator.
    """
    output: list[RetrievalQuestion] = []
    for question in questions:
        output.append(question)
        if question.answerability != "supported" or not question.expected_source_kind:
            continue
        q = question.question
        q_lower = q.casefold()
        variants: list[str] = []
        if "active condition" in q_lower:
            variants = [
                "Which diagnosis is on the active problem list?",
                "What problem was recorded in the encounter summary?",
            ]
        elif "what value is documented for" in q_lower:
            name = re.sub(r"(?i)^what value is documented for\s+", "", q).rstrip(" ?")
            variants = [
                f"Give me the recorded {name} result.",
                f"What measurement was reported for {name}?",
            ]
        elif "which medication" in q_lower:
            variants = [
                "Which drug is listed in medication reconciliation?",
                "What medicine is recorded in the reviewed chart?",
            ]
        for index, variant in enumerate(variants, 1):
            output.append(RetrievalQuestion(
                id=f"{question.id}-p{index}", patient_key=question.patient_key,
                question=variant, answerability=question.answerability,
                expected_source_kind=question.expected_source_kind,
                expected_fact=question.expected_fact,
            ))
    return output

def evaluate_retriever(retriever: Retriever, documents: list[RetrievalDocument], questions: list[RetrievalQuestion]) -> BenchmarkSummary:
    by_patient: dict[str, list[RetrievalDocument]] = {}
    for document in documents:
        by_patient.setdefault(document.patient_key, []).append(document)
    supported = [question for question in questions if question.answerability == "supported" and question.expected_source_kind]
    negatives = [question for question in questions if question.answerability != "supported"]
    hits_at = {1: 0, 3: 0, 5: 0}
    reciprocal_sum = 0.0
    misses = 0
    leakage = 0
    for question in supported:
        scoped = by_patient.get(question.patient_key, [])
        ranking = retriever.rank(question.question, scoped, limit=10)
        allowed_ids = {document.id for document in scoped}
        if any(hit.document_id not in allowed_ids for hit in ranking):
            leakage += 1
        target_rank = None
        for index, hit in enumerate(ranking, 1):
            if hit.source_kind == question.expected_source_kind:
                target_rank = index
                break
        if target_rank is None:
            misses += 1
            continue
        reciprocal_sum += 1.0 / target_rank
        for cutoff in hits_at:
            if target_rank <= cutoff:
                hits_at[cutoff] += 1
    denom = max(1, len(supported))
    return BenchmarkSummary(
        backend=retriever.name,
        supported_questions=len(supported),
        not_documented_questions=len(negatives),
        recall_at_1=round(hits_at[1] / denom, 4),
        recall_at_3=round(hits_at[3] / denom, 4),
        recall_at_5=round(hits_at[5] / denom, 4),
        mrr=round(reciprocal_sum / denom, 4),
        patient_leakage_count=leakage,
        source_kind_misses=misses,
    )


def _normalize_for_compare(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip(" .;:")


def evaluate_fact_coverage(documents: list[RetrievalDocument], questions: list[RetrievalQuestion]) -> dict[str, Any]:
    by_patient_kind: dict[tuple[str, str], list[RetrievalDocument]] = {}
    for doc in documents:
        by_patient_kind.setdefault((doc.patient_key, doc.source_kind), []).append(doc)
    supported_total = 0
    supported_found = 0
    not_documented_total = 0
    unexpected_fact_matches = 0
    by_question_type: dict[str, dict[str, int]] = {}

    def classify(question: RetrievalQuestion) -> tuple[str, str | None]:
        q = question.question.casefold()
        if "active condition" in q:
            return "condition", "condition"
        if "which medication" in q:
            return "medication", "medication"
        if "what value" in q:
            return "measurement", "measurement"
        return "other", None

    cache: dict[str, list[Any]] = {}
    for question in questions:
        category, fact_type = classify(question)
        stats = by_question_type.setdefault(category, {"total": 0, "found": 0})
        stats["total"] += 1
        if question.answerability == "supported" and question.expected_source_kind and fact_type:
            supported_total += 1
            found = False
            for document in by_patient_kind.get((question.patient_key, question.expected_source_kind), []):
                if document.id not in cache:
                    source = EligibleSource(
                        id=document.id, kind="R", title=document.title, body=document.text,
                        version=1, record_date=None, updated_at="synthetic",
                    )
                    cache[document.id] = extract_facts(source, question.patient_key)
                expected = _normalize_for_compare(question.expected_fact)
                for fact in cache[document.id]:
                    if fact.fact_type != fact_type:
                        continue
                    candidates = [fact.value_text, fact.entity_label]
                    candidates.extend(str(value) for value in fact.normalized.values() if isinstance(value, (str, int, float)))
                    haystack = " | ".join(_normalize_for_compare(value) for value in candidates)
                    if expected and (expected in haystack or haystack in expected):
                        found = True
                        break
                if not found and fact_type == "medication" and expected.startswith("no scheduled medication documented"):
                    # Explicit negative medication reconciliation is intentionally
                    # represented as absence rather than inventing a medication entity.
                    found = any(expected in _normalize_for_compare(document.text) for document in by_patient_kind.get((question.patient_key, question.expected_source_kind), []))
                if found:
                    break
            if found:
                supported_found += 1
                stats["found"] += 1
        elif question.answerability != "supported":
            not_documented_total += 1
            # The current negative corpus asks for blood type, a fact type the
            # index deliberately does not extract. This guard catches accidental
            # future promotion of that unrelated value.
            if "blood type" in question.question.casefold():
                for document in by_patient_kind.get((question.patient_key, question.expected_source_kind or ""), []):
                    if re.search(r"\bblood\s+type\b", document.text, re.I):
                        unexpected_fact_matches += 1
    return {
        "extraction_version": "provenance-facts-v3",
        "supported_questions": supported_total,
        "supported_fact_matches": supported_found,
        "supported_fact_coverage": round(supported_found / max(1, supported_total), 4),
        "not_documented_questions": not_documented_total,
        "unexpected_negative_matches": unexpected_fact_matches,
        "by_question_type": by_question_type,
    }


def benchmark_all(root: Path, *, include_semantic: bool = False, rerank: bool = False) -> dict[str, Any]:
    documents, base_questions = load_synthetic_corpus(root)
    questions = expand_benchmark_questions(base_questions)
    results: list[dict[str, Any]] = []
    fts = FTS5Retriever()
    fts_summary = evaluate_retriever(fts, documents, questions)
    results.append(fts_summary.public())

    semantic_status: dict[str, Any] = {"enabled": include_semantic, "available": False}
    if include_semantic:
        dense = SentenceTransformerRetriever()
        semantic_status.update({"available": dense.available, "error": dense.error, "model": dense.model_name_or_path})
        if dense.available:
            dense_summary = evaluate_retriever(dense, documents, questions)
            results.append(dense_summary.public())
            # Evaluate a simple RRF hybrid on the same patient-scoped documents.
            by_patient: dict[str, list[RetrievalDocument]] = {}
            for doc in documents:
                by_patient.setdefault(doc.patient_key, []).append(doc)

            class Hybrid:
                name = "fts5+dense_rrf"
                def rank(self, question: str, scoped: list[RetrievalDocument], limit: int = 10) -> list[RetrievalHit]:
                    merged = reciprocal_rank_fusion([
                        fts.rank(question, scoped, limit=20),
                        dense.rank(question, scoped, limit=20),
                    ], limit=max(limit, 20))
                    if rerank:
                        reranker = CrossEncoderReranker()
                        semantic_status["reranker_available"] = reranker.available
                        semantic_status["reranker_error"] = reranker.error
                        semantic_status["reranker_model"] = reranker.model_name_or_path
                        if reranker.available:
                            return reranker.rerank(question, scoped, merged, limit=limit)
                    return merged[:limit]

            results.append(evaluate_retriever(Hybrid(), documents, questions).public())

    return {
        "revision": RETRIEVAL_REVISION,
        "documents": len(documents),
        "base_questions": len(base_questions),
        "questions": len(questions),
        "fact_coverage": evaluate_fact_coverage(documents, questions),
        "retrieval": results,
        "semantic": semantic_status,
        "policy": {
            "production_default": "fts5",
            "semantic_auto_enabled": False,
            "patient_scope_before_ranking": True,
            "network_downloads_allowed": False,
            "promotion_gate": {
                "minimum_recall_at_5": 0.98,
                "minimum_mrr_delta_over_fts5": 0.01,
                "maximum_patient_leakage_count": 0,
            },
        },
    }


def semantic_promotion_decision(report: dict[str, Any]) -> dict[str, Any]:
    """Return an explicit decision; never auto-enable semantic retrieval."""
    retrieval = list(report.get("retrieval") or [])
    baseline = next((item for item in retrieval if item.get("backend") == "fts5"), None)
    gate = (report.get("policy") or {}).get("promotion_gate") or {}
    candidates = [item for item in retrieval if item.get("backend") != "fts5"]
    if not baseline:
        return {"promote": False, "reason": "missing_fts5_baseline", "candidates": []}
    decisions = []
    for candidate in candidates:
        recall_ok = float(candidate.get("recall_at_5") or 0.0) >= float(gate.get("minimum_recall_at_5") or 0.98)
        leakage_ok = int(candidate.get("patient_leakage_count") or 0) <= int(gate.get("maximum_patient_leakage_count") or 0)
        delta = float(candidate.get("mrr") or 0.0) - float(baseline.get("mrr") or 0.0)
        mrr_ok = delta >= float(gate.get("minimum_mrr_delta_over_fts5") or 0.01)
        decisions.append({
            "backend": candidate.get("backend"),
            "eligible": bool(recall_ok and leakage_ok and mrr_ok),
            "recall_ok": recall_ok,
            "leakage_ok": leakage_ok,
            "mrr_delta": round(delta, 4),
            "mrr_delta_ok": mrr_ok,
        })
    eligible = [item for item in decisions if item["eligible"]]
    return {
        "promote": False,  # human approval is always required
        "automatic_promotion_allowed": False,
        "eligible_candidates": eligible,
        "candidates": decisions,
        "reason": (
            "candidate_meets_metric_gate_but_requires_human_approval"
            if eligible else "no_candidate_beats_the_fts5_gate"
        ),
    }
