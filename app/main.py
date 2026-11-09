"""Local clinical-records prototype. Run ONE worker; no plaintext database files."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import schemas as S
from .assistant_memory import (
    append_exchange,
    clear_thread,
    get_dialogue_state,
    get_owned_thread,
    history_for_thread,
    list_messages,
    list_threads,
    open_thread,
    thread_public,
)
from .chart_assistant import (
    INTENT_LATEST_REPORT,
    INTENT_OPEN,
    INTENT_OVERVIEW,
    answer_looks_like_source_dump,
    deduplicate_sources,
    history_dicts,
    query_terms,
    requested_word_limit,
    resolve_intent,
    sanitize_record_text,
    source_metadata,
    source_score,
    structured_chart_answer,
    truncate_words,
)
from .dialogue_state import (
    apply_requested_format,
    enrich_dialogue_resolution,
    resolution_clarification,
    resolve_dialogue_state,
)
from .demo_records import DEMO_RECORDS
from .intake import MAX_UPLOAD, extract, identity
from .model import LocalModel, ModelUnavailable
from .fact_store import (
    EXTRACTION_VERSION,
    answer_from_facts,
    fact_inventory,
    fact_resolutions_for_patient,
    fact_sources_for_ui,
    facts_for_patient,
    sync_patient_facts,
    sync_source_facts,
    validate_fact_provenance,
    validate_resolution_provenance,
)
from .retrieval_eval import FTS5Retriever, RETRIEVAL_REVISION, RetrievalDocument
from .retrieval_contracts import HYBRID_RETRIEVAL_REVISION
from .decision_models import DECISION_RAG_REVISION, DECISION_RESCUE_REVISION
from .medication_event_workflow import (
    apply_workflow_clarification,
    event_sources_for_ui,
    load_eligible_sources,
    search_medication_event,
    stamp_event_resolution,
    validate_source_snapshot,
)
from .vault import SCHEMA_VERSION, UnsupportedSchemaError, Vault, canonical, now, uid

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
PW = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
DUMMY_HASH = PW.hash(secrets.token_urlsafe(24))
SESSION_SECONDS = 1800


def abort(code: int, message: str):
    raise HTTPException(code, message)


def safe_user(row) -> dict:
    return {k: row[k] for k in ["id", "username", "role", "active"]}


_RECORD_LOOKUP_RE = re.compile(
    r"\b(?:chart|record|records|documented|listed|prescribed|medication\s+list|according\s+to|does\s+the\s+chart|what\s+does\s+the\s+record)\b",
    re.I,
)
_TREATMENT_ADVICE_PATTERNS = [
    re.compile(r"\bwhat\s+(?:can|could|should)\s+(?:i|the\s+patient|they|he|she|we)\s+(?:take|use|do)\b", re.I),
    re.compile(r"\bwhat\s+to\s+take\b", re.I),
    re.compile(r"\b(?:what|which)\s+(?:medicine|medication|drug|treatment)\b.*\b(?:for|should|recommend)\b", re.I),
    re.compile(r"\b(?:recommend|prescribe|best\s+medicine|treatment\s+options?|medication\s+options?)\b", re.I),
    re.compile(r"\bhow\s+much\b.*\b(?:take|dose|dosage)\b", re.I),
]
_AMBIGUOUS_FOLLOW_UP_RE = re.compile(
    r"^\s*(?:what\s+else|what\s+other\s+options(?:\s+are\s+there)?|other\s+options|alternatives?|and\s+what\s+about|what\s+about\s+.+|how\s+much|why)\??\s*$",
    re.I,
)
_SOURCE_HEADER_RE = re.compile(
    r"\[[A-Z]\d+\]\s+[^\n]*?\(version\s+\d+\s*,\s*excerpt\)\.?\s*",
    re.I,
)


def _history_payload(history) -> list[dict]:
    return [turn.model_dump() if hasattr(turn, "model_dump") else dict(turn) for turn in history][-8:]


def _last_user_questions(history: list[dict], limit: int = 2) -> list[str]:
    return [str(turn.get("content") or "").strip() for turn in history if turn.get("role") == "user" and str(turn.get("content") or "").strip()][-limit:]


def _record_lookup_requested(question: str) -> bool:
    return bool(_RECORD_LOOKUP_RE.search(question))


def _direct_treatment_request(question: str) -> bool:
    if _record_lookup_requested(question):
        return False
    return any(pattern.search(question) for pattern in _TREATMENT_ADVICE_PATTERNS)


def treatment_advice_request(question: str, history: list[dict]) -> bool:
    if _direct_treatment_request(question):
        return True
    if _AMBIGUOUS_FOLLOW_UP_RE.match(question):
        return any(_direct_treatment_request(previous) for previous in _last_user_questions(history, 3))
    return False


def conversational_retrieval_query(question: str, history: list[dict]) -> str:
    """Use prior user wording for pronoun/follow-up retrieval, never prior AI answers."""
    previous = _last_user_questions(history, 2)
    return " ".join(previous + [question])[-4500:]


def treatment_boundary_answer(question: str, history: list[dict]) -> str:
    conversation = " ".join(_last_user_questions(history, 3) + [question]).casefold()
    if "fever" in conversation:
        topic = "fever treatment"
    elif "pain" in conversation or "headache" in conversation:
        topic = "pain treatment"
    elif "infection" in conversation:
        topic = "infection treatment"
    else:
        topic = "the requested treatment"
    return (
        "I can summarize what this chart documents, but I can’t choose a medication, dose, or treatment. "
        f"I don’t see a clinician-approved plan for {topic} in the reviewed record. "
        "Please use clinical judgment or a separate general medical reference. I can still list the medications, allergies, and plans recorded in this chart."
    )


def _requested_word_limit(question: str) -> int | None:
    match = re.search(r"\b(?:in|under|about|max(?:imum)?(?:\s+of)?)\s+(\d{1,3})\s+words?\b", question, re.I)
    if not match:
        return None
    return max(5, min(int(match.group(1)), 150))


def _truncate_words(text: str, limit: int | None) -> str:
    if not limit:
        return text.strip()
    words = text.strip().split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).rstrip(" ,;:") + "."


def _clean_source_facts(sources: list[dict]) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()
    for source in sources:
        text = str(source.get("excerpt") or "")
        lines: list[str] = []
        for raw in text.splitlines():
            line = raw.strip().lstrip("-• ")
            if not line:
                continue
            lower = line.casefold()
            if any(phrase in lower for phrase in (
                "synthetic training", "synthetic record", "demo data", "fictional patient", "not a real patient", "for testing only",
                "no treatment recommendation is generated by this software", "software fixture",
            )):
                continue
            if re.match(r"^(?:patient|dob|mrn|report date|document type|source|page)\s*:", line, re.I):
                continue
            lines.append(line)
        joined = " ".join(lines)
        for sentence in re.split(r"(?<=[.!?])\s+", joined):
            fact = " ".join(sentence.split()).strip(" -")
            if len(fact) < 8:
                continue
            key = re.sub(r"\W+", " ", fact.casefold()).strip()
            if key and key not in seen:
                seen.add(key)
                facts.append(fact.rstrip(".") + ".")
    return facts


def _fact_clause(fact: str) -> str:
    text = fact.strip().rstrip(".")
    mappings = [
        (r"^Active problems:\s*", "active problems including "),
        (r"^Current medications:\s*", "current medications including "),
        (r"^Medication allergy:\s*", "a medication allergy: "),
        (r"^Medication allergies:\s*", "medication allergies: "),
        (r"^Documented plan:\s*", "a documented plan to "),
    ]
    for pattern, replacement in mappings:
        if re.search(pattern, text, re.I):
            text = re.sub(pattern, replacement, text, flags=re.I)
            break
    text = re.sub(r"^The reviewed outside summary lists\s+", "", text, flags=re.I)
    text = re.sub(r"\s*;\s*", ", and ", text)
    if text:
        text = text[0].lower() + text[1:]
    return text


def fallback_record_answer(question: str, sources: list[dict]) -> str:
    facts = _clean_source_facts(sources)
    if not facts:
        return "I don’t have enough reviewed information in this chart to answer that."
    lower = question.casefold()
    if re.search(r"\b(?:missing|incomplete|pending|not documented|unknown|still needed)\b", lower):
        preferred = [fact for fact in facts if re.search(r"\b(?:pending|missing|incomplete|not documented|not available|requested|confirm|unknown|outstanding)\b", fact, re.I)]
        chosen = preferred[:1] or facts[:2]
        answer = "The chart notes that " + "; ".join(_fact_clause(fact) for fact in chosen) + "."
    elif re.search(r"\b(?:medication|medications|medicine|drug|allerg|prescription)\b", lower):
        structured = [fact for fact in facts if re.match(r"^(?:Current medications|Medication allerg)", fact, re.I)]
        preferred = structured or [fact for fact in facts if re.search(r"\b(?:medication|tablet|capsule|inhaler|mg\b|allerg|prescri)\b", fact, re.I)]
        if not preferred:
            answer = "I don’t see a reviewed medication or allergy entry in the available chart excerpts."
        else:
            answer = "The chart lists " + "; ".join(_fact_clause(fact) for fact in preferred[:3]) + "."
    else:
        structured = sorted(
            facts,
            key=lambda fact: (
                0 if re.match(r"^(?:Active problems|Current medications|Medication allerg)", fact, re.I) else
                1 if re.search(r"\b(?:pending|outstanding|not yet|has not)\b", fact, re.I) else 2
            ),
        )
        answer = "The chart documents " + "; ".join(_fact_clause(fact) for fact in structured[:4]) + "."
    return _truncate_words(answer, _requested_word_limit(question))


def clean_model_answer(answer: str) -> str:
    text = str(answer or "").strip()
    text = _SOURCE_HEADER_RE.sub("", text)
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.casefold().rstrip(":")
        if not stripped:
            continue
        if lower in {"unverified model draft", "supplied evidence", "answer"}:
            continue
        if re.match(r"^source labels are not verified", lower):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()


def low_quality_answer(answer: str) -> bool:
    if not answer:
        return True
    words = re.findall(r"[A-Za-z0-9']+", answer)
    if len(words) < 7:
        return True
    without_labels = re.sub(r"\[[A-Z]\d+\]", "", answer).strip(" .,:;\n")
    return len(re.findall(r"[A-Za-z0-9']+", without_labels)) < 6


def unsafe_treatment_language(answer: str) -> bool:
    return bool(re.search(
        r"\b(?:you|the\s+patient|patient|user)\s+(?:should|can|could|must)\s+(?:take|use|start|stop)\b|\btake\s+[A-Za-z][A-Za-z0-9-]*\s+for\s+(?:a\s+)?(?:fever|pain|infection)",
        answer,
        re.I,
    ))


def create_app(data_dir: str | Path | None = None, model_client=None) -> FastAPI:
    vault = Vault(Path(data_dir or os.getenv("DATA_DIR", ROOT / "data")))
    model = model_client or LocalModel()
    sessions: dict[str, dict] = {}
    attempts = defaultdict(deque)
    idle_seconds = int(os.getenv("VAULT_IDLE_SECONDS", "900"))
    origins = set(os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(","))

    async def auto_lock():
        while True:
            await asyncio.sleep(5)
            with vault.mutex:
                if vault.unlocked and time.monotonic() - vault.last_activity > idle_seconds:
                    with vault.transaction():
                        vault.audit("system", None, "vault.auto_lock", "vault", {})
                    vault.lock()
                    sessions.clear()

    @asynccontextmanager
    async def lifespan(_app):
        vault.acquire_process_lock()
        task = asyncio.create_task(auto_lock())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        vault.close()
        if hasattr(model, "close"):
            model.close()

    app = FastAPI(title="Clinical Workspace — Stage 2 Encrypted Assistant Threads", version="0.2.4", lifespan=lifespan,
                  docs_url=None, openapi_url="/api/openapi.json", redoc_url=None)
    app.state.vault, app.state.model = vault, model
    hosts = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if data_dir is not None:
        hosts.append("testserver")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in origins:
            return JSONResponse({"detail": "Cross-origin access is blocked."}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site access is blocked."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
            "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic's default error response can echo secret/PHI input values.
        return JSONResponse({"detail": "Invalid request fields.", "errors": [
            {"loc": list(e["loc"]), "type": e["type"], "message": e["msg"]} for e in exc.errors()
        ]}, status_code=422)

    @app.exception_handler(Exception)
    async def unknown_error(request, exc):
        # Deliberately do not log exception arguments or request bodies.
        return JSONResponse({"detail": "Operation failed; no success was recorded. Check local configuration."}, status_code=500)

    def throttle(request: Request, kind: str):
        key = (request.client.host if request.client else "local", kind)
        timestamp = time.monotonic()
        q = attempts[key]
        while q and q[0] < timestamp - 300:
            q.popleft()
        if len(q) >= 12:
            abort(429, "Too many authentication attempts. Try again in five minutes.")
        q.append(timestamp)

    def authenticated(request: Request) -> dict:
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        with vault.mutex:
            if not vault.unlocked:
                abort(423, "Vault is locked. Unlock it first.")
            session = sessions.get(token)
            timestamp = time.monotonic()
            if not session or session["expires"] < timestamp or session["generation"] != vault.generation:
                sessions.pop(token, None)
                abort(401, "Sign in again.")
            if timestamp - vault.last_activity > idle_seconds:
                vault.lock(); sessions.clear()
                abort(423, "Vault locked after inactivity.")
            row = vault.db.execute("SELECT * FROM users WHERE id=? AND active=1", (session["user_id"],)).fetchone()
            if not row:
                abort(401, "Account unavailable.")
            vault.last_activity = timestamp
            return safe_user(row)

    def admin(user):
        if user["role"] != "admin":
            abort(403, "Administrator permission required.")

    def clinician(user):
        if user["role"] not in {"doctor", "admin"}:
            abort(403, "Doctor permission required for clinical edits, approval and assistant use.")

    def patient_access(user, patient_id):
        row = vault.db.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
        grant = user["role"] == "admin" or vault.db.execute(
            "SELECT 1 FROM grants WHERE patient_id=? AND user_id=?", (patient_id, user["id"])).fetchone()
        if not row or not grant:
            abort(404, "Patient not found or not assigned to your account.")
        return dict(row)

    def session_access(user, session_id, patient_id=None):
        row = vault.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            abort(404, "Session not found or unavailable.")
        patient_access(user, row["patient_id"])
        if patient_id is not None and row["patient_id"] != patient_id:
            abort(422, "Session does not belong to the selected patient.")
        return row

    def doc_access(user, document_id):
        row = vault.db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            abort(404, "Document not found or unavailable.")
        if row["patient_id"]:
            patient_access(user, row["patient_id"])
        elif user["role"] != "admin" and row["created_by"] != user["id"]:
            abort(404, "Document not found or unavailable.")
        return row

    def bump(patient_id):
        if patient_id:
            vault.db.execute("UPDATE patients SET revision=revision+1, updated_at=? WHERE id=?", (now(), patient_id))

    def session_public(row):
        return {key: row[key] for key in [
            "id", "patient_id", "title", "session_type", "status", "starts_at", "ends_at",
            "location", "clinician_id", "created_by", "version", "created_at", "updated_at"
        ]}

    def document_public(row, full=False):
        keys = ["id", "patient_id", "session_id", "filename", "mime", "sha256", "summary", "summary_kind", "method", "state",
                "context_enabled", "version", "created_by", "created_at", "updated_at"]
        result = {k: row[k] for k in keys}
        result["identity"] = json.loads(row["identity_json"])
        if full:
            result["text"] = row["text"]
        return result

    def assistant_thread_response(db, row, current_revision, context_reset=False):
        messages, truncated = list_messages(
            db,
            thread_id=row["id"],
            current_revision=current_revision,
        )
        count = db.execute(
            "SELECT COUNT(*) FROM assistant_messages WHERE thread_id=?", (row["id"],)
        ).fetchone()[0]
        result = thread_public(
            row,
            message_count=count,
            context_reset=context_reset,
        )
        result["messages"] = messages
        result["messages_truncated"] = truncated
        result["evidence_scope"] = "longitudinal"
        result["dialogue_state"] = get_dialogue_state(
            db,
            thread_id=row["id"],
            current_revision=current_revision,
        )
        return result

    @app.get("/api/status")
    def status():
        with vault.mutex:
            active_schema = vault.schema_version() if vault.unlocked else None
            migrations = list(vault.last_migrations)
        return {"setup_required": not vault.exists, "vault_unlocked": vault.unlocked, "stage": 2,
                "stage2_increment": "provenance-conflict-semantics", "retrieval_evaluation_increment": "confidence-gated-decision-rescue", "assistant_revision": "dialogue-state-v1", "workflow_revision": EXTRACTION_VERSION, "fact_semantics_revision": "fact-semantics-v1", "retrieval_revision": RETRIEVAL_REVISION, "hybrid_retrieval_revision": HYBRID_RETRIEVAL_REVISION, "decision_rag_revision": DECISION_RAG_REVISION, "decision_rescue_revision": DECISION_RESCUE_REVISION, "decision_retrieval_evaluation_only": True, "decision_rescue_evaluation_only": True, "decision_retrieval_enabled": False, "remote_decision_model_egress_allowed": False, "retrieval_default": "patient_scoped_fts5+intent_score", "semantic_retrieval_enabled": False, "hybrid_retrieval_evaluation_only": True, "remote_embedding_egress_allowed": False, "remote_reranking_egress_allowed": False, "persistent_vector_store_enabled": False, "model_runtime_revision": "runtime-budget-v1", "schema_version": active_schema,
                "target_schema_version": SCHEMA_VERSION, "migrations": migrations,
                "synthetic_only": True, "model_endpoint": model.base_url, "model_name": model.model,
                "remote_inference": model.remote, "internet_research_enabled": False,
                "notice": "Patient-scoped FTS5 with deterministic clinical query expansion remains the live default. Confidence-gated local BGE/Arc rescue is evaluation-only: strong FTS queries bypass neural retrieval, weak queries may be tested locally, and no strategy can auto-promote."}

    @app.post("/api/setup")
    def setup(body: S.Setup, request: Request):
        throttle(request, "unlock")
        with vault.mutex:
            if vault.exists or vault.unlocked:
                abort(409, "Workspace already initialized.")
            if body.password == body.vault_password:
                abort(422, "Use separate vault and user passwords.")
            vault.create(body.vault_password)
            try:
                with vault.transaction() as db:
                    user_id = uid()
                    db.execute("INSERT INTO users(id,username,role,password_hash,created_at) VALUES(?,?,?,?,?)",
                               (user_id, body.username.lower(), "admin", PW.hash(body.password), now()))
                    vault.audit(user_id, None, "workspace.created", "vault", {"first_account": body.username.lower()})
            except Exception:
                vault.lock()
                raise
        return {"created": True, "next": "Sign in with your individual account."}

    @app.post("/api/unlock")
    def unlock(body: S.PasswordBody, request: Request):
        throttle(request, "unlock")
        with vault.mutex:
            if vault.unlocked:
                abort(409, "Vault is already unlocked. Sign in with your individual account.")
            if not vault.exists:
                abort(409, "Create the workspace first.")
            try:
                vault.unlock(body.password)
            except UnsupportedSchemaError as exc:
                abort(409, str(exc))
            except Exception:
                abort(401, "Unable to unlock: incorrect password or damaged vault.")
            with vault.transaction():
                vault.audit("vault_operator", None, "vault.unlocked", "vault", {"migrations": vault.last_migrations})
        return {"unlocked": True, "schema_version": vault.schema_version(), "migrations": vault.last_migrations}

    @app.post("/api/login")
    def login(body: S.Login, request: Request):
        throttle(request, "login")
        with vault.mutex:
            if not vault.unlocked:
                abort(423, "Unlock the workspace first.")
            row = vault.db.execute("SELECT * FROM users WHERE username=?", (body.username.lower(),)).fetchone()
            try:
                PW.verify(row["password_hash"] if row else DUMMY_HASH, body.password)
                valid = row is not None and bool(row["active"])
            except VerificationError:
                valid = False
            if not valid:
                with vault.transaction():
                    vault.audit("anonymous", None, "auth.failed", "login", {})
                abort(401, "Invalid credentials.")
            token = secrets.token_urlsafe(32)
            sessions[token] = {"user_id": row["id"], "expires": time.monotonic() + SESSION_SECONDS,
                               "generation": vault.generation}
            vault.last_activity = time.monotonic()
            with vault.transaction():
                vault.audit(row["id"], None, "auth.login", row["id"], {})
            return {"token": token, "user": safe_user(row), "expires_in": SESSION_SECONDS}

    @app.get("/api/me")
    def me(user=Depends(authenticated)):
        return user

    @app.post("/api/logout")
    def logout(request: Request, user=Depends(authenticated)):
        with vault.transaction():
            vault.audit(user["id"], None, "auth.logout", user["id"], {})
            sessions.pop(request.headers.get("authorization", "").removeprefix("Bearer "), None)
        return {"logged_out": True}

    @app.post("/api/lock")
    def lock(user=Depends(authenticated)):
        admin(user)
        with vault.mutex:
            with vault.transaction():
                vault.audit(user["id"], None, "vault.locked", "vault", {})
            sessions.clear(); vault.lock()
        return {"locked": True}

    @app.get("/api/model/status")
    def model_status(user=Depends(authenticated)):
        return model.status()

    @app.get("/api/users")
    def users(user=Depends(authenticated)):
        admin(user)
        with vault.mutex:
            return [safe_user(r) for r in vault.db.execute("SELECT * FROM users ORDER BY username")]

    @app.post("/api/users", status_code=201)
    def user_create(body: S.UserCreate, user=Depends(authenticated)):
        admin(user)
        with vault.transaction() as db:
            user_id = uid()
            try:
                db.execute("INSERT INTO users(id,username,role,password_hash,created_at) VALUES(?,?,?,?,?)",
                           (user_id, body.username.lower(), body.role, PW.hash(body.password), now()))
            except sqlite3.IntegrityError:
                abort(409, "Username already exists.")
            vault.audit(user["id"], None, "user.created", user_id, {"username": body.username.lower(), "role": body.role})
        return {"id": user_id, "username": body.username.lower(), "role": body.role, "active": 1}

    @app.patch("/api/users/{user_id}")
    def user_active(user_id: str, body: S.UserActive, user=Depends(authenticated)):
        admin(user)
        if user_id == user["id"]:
            abort(422, "Cannot deactivate the sole administrator.")
        with vault.transaction() as db:
            row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            if not row: abort(404, "Account not found.")
            db.execute("UPDATE users SET active=? WHERE id=?", (int(body.active), user_id))
            vault.audit(user["id"], None, "user.access_changed", user_id, {"before": bool(row["active"]), "after": body.active, "reason": body.reason})
            for token in [k for k,v in sessions.items() if v["user_id"] == user_id]:
                sessions.pop(token, None)
        return {"updated": True}

    @app.get("/api/patients")
    def patients(user=Depends(authenticated)):
        with vault.transaction() as db:
            if user["role"] == "admin":
                rows = db.execute("SELECT * FROM patients ORDER BY updated_at DESC").fetchall()
            else:
                rows = db.execute("SELECT p.* FROM patients p JOIN grants g ON g.patient_id=p.id WHERE g.user_id=? ORDER BY p.updated_at DESC", (user["id"],)).fetchall()
            result = []
            for row in rows:
                obj = dict(row)
                obj["document_count"] = db.execute("SELECT COUNT(*) FROM documents WHERE patient_id=?", (row["id"],)).fetchone()[0]
                obj["review_count"] = db.execute("SELECT COUNT(*) FROM documents WHERE patient_id=? AND state='review'", (row["id"],)).fetchone()[0]
                obj["session_count"] = db.execute("SELECT COUNT(*) FROM sessions WHERE patient_id=?", (row["id"],)).fetchone()[0]
                result.append(obj)
            vault.audit(user["id"], None, "patients.listed", "patients", {"count": len(result)})
            return result

    def insert_patient(body, user):
        patient_id = uid()
        timestamp = now()
        try:
            vault.db.execute("INSERT INTO patients(id,name,dob,mrn,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                             (patient_id, body.name, body.dob.isoformat(), body.mrn.upper(), timestamp, timestamp))
        except sqlite3.IntegrityError:
            abort(409, "MRN already exists. Review the existing patient; names are not unique identifiers.")
        vault.db.execute("INSERT INTO grants(patient_id,user_id) VALUES(?,?)", (patient_id, user["id"]))
        vault.audit(user["id"], patient_id, "patient.created", patient_id,
                    {"name": body.name, "dob": body.dob.isoformat(), "mrn": body.mrn.upper(), "identity_confirmed_by_user": True})
        return patient_id

    @app.post("/api/patients", status_code=201)
    def patient_create(body: S.PatientCreate, user=Depends(authenticated)):
        with vault.transaction():
            patient_id = insert_patient(body, user)
            return dict(vault.db.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone())

    @app.get("/api/patients/{patient_id}")
    def patient_detail(patient_id: str, user=Depends(authenticated)):
        with vault.transaction() as db:
            patient = patient_access(user, patient_id)
            documents = [document_public(r) for r in db.execute("SELECT * FROM documents WHERE patient_id=? ORDER BY created_at DESC", (patient_id,))]
            notes = [dict(r) for r in db.execute("SELECT n.*,u.username AS author FROM notes n JOIN users u ON u.id=n.created_by WHERE patient_id=? ORDER BY updated_at DESC", (patient_id,))]
            patient_sessions = [dict(r) for r in db.execute(
                "SELECT s.*,u.username AS clinician_name FROM sessions s "
                "LEFT JOIN users u ON u.id=s.clinician_id WHERE s.patient_id=? ORDER BY s.starts_at DESC",
                (patient_id,),
            )]
            vault.audit(user["id"], patient_id, "patient.viewed", patient_id, {})
            return {"patient": patient, "documents": documents, "notes": notes, "sessions": patient_sessions}

    @app.patch("/api/patients/{patient_id}")
    def patient_update(patient_id: str, body: S.PatientUpdate, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            before = patient_access(user, patient_id)
            if before["revision"] != body.expected_revision:
                abort(409, "Patient changed. Refresh before editing.")
            try:
                db.execute("UPDATE patients SET name=?,dob=?,mrn=?,updated_at=?,revision=revision+1 WHERE id=?",
                           (body.name, body.dob.isoformat(), body.mrn.upper(), now(), patient_id))
            except sqlite3.IntegrityError:
                abort(409, "That MRN already belongs to another chart.")
            vault.audit(user["id"], patient_id, "patient.corrected", patient_id,
                        {"before": before, "after": body.model_dump(mode="json"), "reason": body.reason})
        return {"updated": True}

    @app.get("/api/patients/{patient_id}/sessions")
    def patient_sessions(patient_id: str, user=Depends(authenticated)):
        with vault.transaction() as db:
            patient_access(user, patient_id)
            rows = db.execute(
                "SELECT s.*,u.username AS clinician_name FROM sessions s "
                "LEFT JOIN users u ON u.id=s.clinician_id WHERE s.patient_id=? ORDER BY s.starts_at DESC",
                (patient_id,),
            ).fetchall()
            vault.audit(user["id"], patient_id, "sessions.listed", patient_id, {"count": len(rows)})
            return [dict(row) for row in rows]

    @app.post("/api/patients/{patient_id}/sessions", status_code=201)
    def session_create(patient_id: str, body: S.SessionCreate, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            patient_access(user, patient_id)
            clinician_id = body.clinician_id or user["id"]
            clinician_row = db.execute(
                "SELECT id,role FROM users WHERE id=? AND active=1", (clinician_id,)
            ).fetchone()
            if not clinician_row or clinician_row["role"] not in {"doctor", "admin"}:
                abort(422, "Selected clinician must be an active doctor or administrator.")
            if clinician_row["role"] == "doctor" and not db.execute(
                "SELECT 1 FROM grants WHERE patient_id=? AND user_id=?", (patient_id, clinician_id)
            ).fetchone():
                abort(422, "Selected doctor must be assigned to this patient before the session is created.")
            session_id = uid()
            timestamp = now()
            db.execute(
                """INSERT INTO sessions(
                    id,patient_id,title,session_type,status,starts_at,ends_at,location,
                    clinician_id,created_by,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    session_id, patient_id, body.title, body.session_type, body.status,
                    body.starts_at.isoformat(), body.ends_at.isoformat() if body.ends_at else None,
                    body.location, clinician_id, user["id"], timestamp, timestamp,
                ),
            )
            bump(patient_id)
            vault.audit(user["id"], patient_id, "session.created", session_id, {
                "title": body.title,
                "session_type": body.session_type,
                "status": body.status,
                "starts_at": body.starts_at.isoformat(),
                "ends_at": body.ends_at.isoformat() if body.ends_at else None,
                "location": body.location,
                "clinician_id": clinician_id,
            })
            return session_public(db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone())

    @app.get("/api/sessions/{session_id}")
    def session_detail(session_id: str, user=Depends(authenticated)):
        with vault.transaction() as db:
            row = session_access(user, session_id)
            result = session_public(row)
            clinician_row = db.execute("SELECT username FROM users WHERE id=?", (row["clinician_id"],)).fetchone() if row["clinician_id"] else None
            result["clinician_name"] = clinician_row[0] if clinician_row else None
            result["document_count"] = db.execute("SELECT COUNT(*) FROM documents WHERE session_id=?", (session_id,)).fetchone()[0]
            result["note_count"] = db.execute("SELECT COUNT(*) FROM notes WHERE session_id=?", (session_id,)).fetchone()[0]
            vault.audit(user["id"], row["patient_id"], "session.viewed", session_id, {})
            return result

    @app.patch("/api/sessions/{session_id}")
    def session_update(session_id: str, body: S.SessionUpdate, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            row = session_access(user, session_id)
            before = session_public(row)
            if row["version"] != body.expected_version:
                abort(409, "Session changed. Refresh before editing.")
            clinician_id = body.clinician_id or row["clinician_id"] or user["id"]
            clinician_row = db.execute(
                "SELECT id,role FROM users WHERE id=? AND active=1", (clinician_id,)
            ).fetchone()
            if not clinician_row or clinician_row["role"] not in {"doctor", "admin"}:
                abort(422, "Selected clinician must be an active doctor or administrator.")
            if clinician_row["role"] == "doctor" and not db.execute(
                "SELECT 1 FROM grants WHERE patient_id=? AND user_id=?", (row["patient_id"], clinician_id)
            ).fetchone():
                abort(422, "Selected doctor must be assigned to this patient before the session is updated.")
            timestamp = now()
            db.execute(
                """UPDATE sessions SET title=?,session_type=?,status=?,starts_at=?,ends_at=?,location=?,
                   clinician_id=?,version=version+1,updated_at=? WHERE id=?""",
                (
                    body.title, body.session_type, body.status, body.starts_at.isoformat(),
                    body.ends_at.isoformat() if body.ends_at else None, body.location,
                    clinician_id, timestamp, session_id,
                ),
            )
            bump(row["patient_id"])
            after = session_public(db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone())
            vault.audit(user["id"], row["patient_id"], "session.updated", session_id, {
                "before": before,
                "after": after,
                "reason": body.reason,
            })
            return after

    @app.get("/api/patients/{patient_id}/access")
    def patient_grants(patient_id: str, user=Depends(authenticated)):
        admin(user)
        with vault.mutex:
            patient_access(user, patient_id)
            return [dict(r) for r in vault.db.execute("SELECT u.id,u.username,u.role FROM grants g JOIN users u ON u.id=g.user_id WHERE g.patient_id=?", (patient_id,))]

    @app.post("/api/patients/{patient_id}/access")
    def grant(patient_id: str, body: S.Assignment, user=Depends(authenticated)):
        admin(user)
        with vault.transaction() as db:
            patient_access(user, patient_id)
            target = db.execute("SELECT id FROM users WHERE id=?", (body.user_id,)).fetchone()
            if not target: abort(404, "Account not found.")
            if body.enabled:
                db.execute("INSERT OR IGNORE INTO grants(patient_id,user_id) VALUES(?,?)", (patient_id, body.user_id))
            else:
                db.execute("DELETE FROM grants WHERE patient_id=? AND user_id=?", (patient_id, body.user_id))
            vault.audit(user["id"], patient_id, "patient.access_granted" if body.enabled else "patient.access_revoked", body.user_id, {"reason": body.reason})
        return {"updated": True}

    def ingest(raw, filename, use_model, user):
        generation = vault.generation
        extraction = extract(filename, raw, model, use_model)
        with vault.transaction() as db:
            if generation != vault.generation:
                abort(409, "Vault changed during extraction. Upload again after signing in.")
            live = db.execute("SELECT active FROM users WHERE id=?", (user["id"],)).fetchone()
            if not live or not live["active"]:
                abort(403, "Account no longer active.")
            detected = extraction.identity
            patient_id = None
            routing = "Needs identity review. Create or select the correct patient; names alone are never sufficient."
            if not detected["ambiguous"] and all(detected.get(k) for k in ("mrn", "dob", "name")) and extraction.method in {"text", "pdf_text"}:
                candidate = db.execute("SELECT * FROM patients WHERE mrn=? AND dob=?", (detected["mrn"].upper(), detected["dob"])).fetchone()
                if candidate and " ".join(candidate["name"].casefold().split()) == " ".join(detected["name"].casefold().split()):
                    allowed = user["role"] == "admin" or db.execute("SELECT 1 FROM grants WHERE patient_id=? AND user_id=?", (candidate["id"],user["id"])).fetchone()
                    if allowed:
                        patient_id = candidate["id"]
                        routing = "Auto-attached using exact MRN + DOB + name. Not in model context until a doctor approves."
            digest = hashlib.sha256(raw).hexdigest()
            # Limit duplicate detection to records this uploader is already allowed to see.
            prior = db.execute("SELECT * FROM documents WHERE sha256=? AND created_by=? AND COALESCE(patient_id,'')=COALESCE(?, '') LIMIT 1", (digest,user["id"],patient_id)).fetchone()
            if prior:
                vault.audit(user["id"], patient_id, "document.duplicate_ignored", prior["id"], {"sha256": digest})
                return {"document": document_public(prior, True), "duplicate": True, "routing": "Identical upload already exists.", "warnings": extraction.warnings}
            doc_id = uid(); timestamp = now()
            db.execute("""INSERT INTO documents(id,patient_id,filename,mime,original,sha256,text,summary,summary_kind,method,identity_json,created_by,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (doc_id,patient_id,filename,extraction.mime,raw,digest,extraction.text,extraction.summary,
                extraction.summary_kind,extraction.method,canonical(detected),user["id"],timestamp,timestamp))
            bump(patient_id)
            vault.audit(user["id"], patient_id, "document.uploaded", doc_id,
                        {"filename": filename, "sha256": digest, "method": extraction.method, "routing": routing,
                         "summary_kind": extraction.summary_kind, "warnings": extraction.warnings})
            row = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return {"document": document_public(row, True), "duplicate": False, "routing": routing, "warnings": extraction.warnings}

    @app.post("/api/intake", status_code=201)
    async def upload(request: Request, user=Depends(authenticated)):
        filename = Path(unquote(request.headers.get("x-filename", "report.txt"))).name[:180]
        if not filename or any(ord(c)<32 for c in filename): abort(422, "Invalid filename.")
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw)+len(chunk) > MAX_UPLOAD: abort(413, "Maximum upload size is 8 MiB.")
            raw.extend(chunk)
        use_model = request.headers.get("x-use-model", "false").lower() == "true"
        try:
            return await run_in_threadpool(ingest, bytes(raw), filename, use_model, user)
        except ValueError as exc:
            abort(422, str(exc))

    @app.get("/api/intake")
    def inbox(user=Depends(authenticated)):
        with vault.transaction() as db:
            if user["role"] == "admin":
                rows = db.execute("SELECT * FROM documents WHERE state='review' ORDER BY created_at DESC").fetchall()
            else:
                rows = db.execute("""SELECT d.* FROM documents d WHERE d.state='review' AND
                    ((d.patient_id IS NULL AND d.created_by=?) OR d.patient_id IN (SELECT patient_id FROM grants WHERE user_id=?))
                    ORDER BY d.created_at DESC""", (user["id"],user["id"])).fetchall()
            vault.audit(user["id"], None, "intake.viewed", "inbox", {"count": len(rows)})
            return [document_public(r) for r in rows]

    @app.get("/api/documents/{document_id}")
    def document_detail(document_id: str, user=Depends(authenticated)):
        with vault.transaction():
            row = doc_access(user, document_id)
            vault.audit(user["id"], row["patient_id"], "document.viewed", document_id, {})
            return document_public(row, True)

    @app.get("/api/documents/{document_id}/file")
    def download_original(document_id: str, user=Depends(authenticated)):
        with vault.transaction():
            row = doc_access(user, document_id)
            vault.audit(user["id"], row["patient_id"], "document.original_downloaded", document_id, {})
            # Force attachment; do not render uploaded PDFs/HTML in the application origin.
            return Response(bytes(row["original"]), media_type="application/octet-stream",
                            headers={"Content-Disposition": f'attachment; filename="original-{document_id}.bin"'})

    @app.post("/api/documents/{document_id}/review")
    def review(document_id: str, body: S.ReviewDocument, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            row = doc_access(user, document_id)
            patient_access(user, body.patient_id)
            if row["version"] != body.expected_version: abort(409, "Document changed. Refresh before reviewing.")
            before = document_public(row, True)
            old_patient = row["patient_id"]
            db.execute("UPDATE documents SET patient_id=?,text=?,summary=?,summary_kind='human_reviewed',state='approved',context_enabled=1,version=version+1,updated_at=? WHERE id=?",
                       (body.patient_id,body.text,body.summary,now(),document_id))
            bump(body.patient_id)
            if old_patient and old_patient != body.patient_id:
                bump(old_patient)
                vault.audit(user["id"], old_patient, "document.reassigned_out", document_id,
                            {"reason": body.reason, "old_patient_id": old_patient, "new_patient_id": body.patient_id})
            fact_sync = sync_source_facts(db, "document", document_id)
            vault.audit(user["id"], body.patient_id, "document.reviewed", document_id,
                        {"before": before, "after": {"text": body.text, "summary": body.summary, "patient_id": body.patient_id},
                         "reason": body.reason, "identity_confirmed": True, "content_confirmed": True})
            vault.audit(user["id"], body.patient_id, "facts.source_synced", document_id,
                        {"source_kind": "document", "fact_count": fact_sync.get("fact_count", 0), "indexed": fact_sync.get("indexed", False)})
            return document_public(db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone(), True)

    @app.post("/api/documents/{document_id}/context")
    def document_context(document_id: str, body: S.ContextChange, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            row = doc_access(user, document_id)
            if row["version"] != body.expected_version: abort(409, "Document changed. Refresh first.")
            if row["state"] != "approved" or not row["patient_id"]:
                abort(422, "Doctor review is required before changing model context.")
            db.execute("UPDATE documents SET context_enabled=?,version=version+1,updated_at=? WHERE id=?", (int(body.enabled),now(),document_id))
            bump(row["patient_id"])
            sync_source_facts(db, "document", document_id)
            vault.audit(user["id"], row["patient_id"], "document.context_included" if body.enabled else "document.context_excluded", document_id,
                        {"before": bool(row["context_enabled"]), "after": body.enabled, "reason": body.reason,
                         "retention": "Original retained; excluded from future retrieval, not destructively erased."})
        return {"updated": True}

    @app.post("/api/patients/{patient_id}/notes", status_code=201)
    def note_create(patient_id: str, body: S.NoteCreate, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            patient_access(user, patient_id)
            note_id = uid(); timestamp = now()
            db.execute("INSERT INTO notes(id,patient_id,title,body,created_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                       (note_id,patient_id,body.title,body.body,user["id"],timestamp,timestamp))
            db.execute("INSERT INTO note_versions VALUES(?,?,?,?,?,?,?)", (note_id,1,body.title,body.body,user["id"],"Initial note",timestamp))
            bump(patient_id)
            fact_sync = sync_source_facts(db, "note", note_id)
            vault.audit(user["id"], patient_id, "note.created", note_id, {"title": body.title, "body": body.body})
            vault.audit(user["id"], patient_id, "facts.source_synced", note_id,
                        {"source_kind": "note", "fact_count": fact_sync.get("fact_count", 0), "indexed": fact_sync.get("indexed", False)})
        return {"id": note_id, "version": 1}

    @app.patch("/api/notes/{note_id}")
    def note_update(note_id: str, body: S.NoteUpdate, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            row = db.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
            if not row: abort(404, "Note unavailable.")
            patient_access(user, row["patient_id"])
            if row["version"] != body.expected_version: abort(409, "Note changed. Refresh to avoid overwriting another edit.")
            version = row["version"] + 1; timestamp = now()
            db.execute("UPDATE notes SET title=?,body=?,version=?,updated_at=? WHERE id=?", (body.title,body.body,version,timestamp,note_id))
            db.execute("INSERT INTO note_versions VALUES(?,?,?,?,?,?,?)", (note_id,version,body.title,body.body,user["id"],body.reason,timestamp))
            bump(row["patient_id"])
            sync_source_facts(db, "note", note_id)
            vault.audit(user["id"], row["patient_id"], "note.edited", note_id,
                        {"before": dict(row), "after": {"title": body.title,"body": body.body,"version": version}, "reason": body.reason})
        return {"updated": True, "version": version}

    @app.get("/api/notes/{note_id}/versions")
    def note_versions(note_id: str, user=Depends(authenticated)):
        with vault.transaction() as db:
            row = db.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
            if not row: abort(404, "Note unavailable.")
            patient_access(user, row["patient_id"])
            vault.audit(user["id"], row["patient_id"], "note.versions_viewed", note_id, {})
            return [dict(r) for r in db.execute("SELECT v.*,u.username AS editor_name FROM note_versions v JOIN users u ON u.id=v.editor WHERE note_id=? ORDER BY version DESC", (note_id,))]

    @app.post("/api/notes/{note_id}/context")
    def note_context(note_id: str, body: S.ContextChange, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            row = db.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
            if not row: abort(404, "Note unavailable.")
            patient_access(user, row["patient_id"])
            if row["version"] != body.expected_version: abort(409, "Note changed. Refresh first.")
            version = row["version"]+1; timestamp = now()
            db.execute("UPDATE notes SET context_enabled=?,version=?,updated_at=? WHERE id=?", (int(body.enabled),version,timestamp,note_id))
            db.execute("INSERT INTO note_versions VALUES(?,?,?,?,?,?,?)", (note_id,version,row["title"],row["body"],user["id"],body.reason,timestamp))
            bump(row["patient_id"])
            sync_source_facts(db, "note", note_id)
            vault.audit(user["id"], row["patient_id"], "note.context_included" if body.enabled else "note.context_excluded", note_id,
                        {"before": bool(row["context_enabled"]), "after": body.enabled, "reason": body.reason})
        return {"updated": True}

    @app.get("/api/patients/{patient_id}/facts")
    def patient_facts(patient_id: str, fact_type: str | None = None, entity: str | None = None, event_type: str | None = None, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            patient_access(user, patient_id)
            inventory = fact_inventory(db, patient_id)
            facts = facts_for_patient(db, patient_id, fact_type=fact_type, entity=entity, event_type=event_type)
            # Every exposed fact must still match the current reviewed source.
            if facts and not validate_fact_provenance(db, facts):
                sync_patient_facts(db, patient_id)
                facts = facts_for_patient(db, patient_id, fact_type=fact_type, entity=entity, event_type=event_type)
                if facts and not validate_fact_provenance(db, facts):
                    abort(409, "Fact index no longer matches the current reviewed sources. Refresh and retry.")
            vault.audit(user["id"], patient_id, "facts.viewed", patient_id,
                        {"fact_type": fact_type, "entity": entity, "event_type": event_type, "count": len(facts), "fact_content_in_audit": False})
            return {"inventory": inventory, "facts": [fact.public() for fact in facts]}

    @app.get("/api/patients/{patient_id}/facts/resolutions")
    def patient_fact_resolutions(
        patient_id: str, fact_type: str | None = None, entity: str | None = None,
        event_type: str | None = None, status: str | None = None, user=Depends(authenticated),
    ):
        clinician(user)
        with vault.transaction() as db:
            patient_access(user, patient_id)
            resolutions = fact_resolutions_for_patient(
                db, patient_id, fact_type=fact_type, entity=entity, event_type=event_type, status=status
            )
            if resolutions and not validate_resolution_provenance(db, resolutions):
                abort(409, "Fact resolution no longer matches the current reviewed source provenance. Refresh and retry.")
            vault.audit(
                user["id"], patient_id, "facts.resolutions_viewed", patient_id,
                {
                    "fact_type": fact_type, "entity": entity, "event_type": event_type,
                    "status": status, "count": len(resolutions), "fact_content_in_audit": False,
                },
            )
            return {"resolutions": [resolution.public() for resolution in resolutions]}

    @app.get("/api/patients/{patient_id}/audit")
    def audit_history(patient_id: str, user=Depends(authenticated)):
        with vault.transaction() as db:
            patient_access(user, patient_id)
            vault.audit(user["id"], patient_id, "audit.viewed", patient_id, {})
            rows = db.execute("SELECT a.*,COALESCE(u.username,a.actor) AS actor_name FROM audit a LEFT JOIN users u ON u.id=a.actor WHERE a.patient_id=? ORDER BY a.seq DESC", (patient_id,)).fetchall()
            return [{**dict(r), "details": json.loads(r["details"])} for r in rows]

    @app.get("/api/audit/verify")
    def audit_verify(user=Depends(authenticated)):
        admin(user)
        with vault.mutex:
            return vault.verify_audit()

    @app.get("/api/audit")
    def workspace_audit(user=Depends(authenticated)):
        admin(user)
        with vault.transaction() as db:
            vault.audit(user["id"], None, "audit.workspace_viewed", "audit", {})
            return [{**dict(r), "details": json.loads(r["details"])} for r in db.execute(
                "SELECT a.*,COALESCE(u.username,a.actor) AS actor_name FROM audit a LEFT JOIN users u ON u.id=a.actor ORDER BY seq DESC")]

    def context_for(user, patient_id, question="", history=None, intent=None, dialogue=None):
        """Return a small, relevant, de-duplicated set of reviewed chart excerpts."""
        patient = patient_access(user, patient_id)
        history = history or []
        resolved_intent = intent or resolve_intent(question, history)
        terms = query_terms(question, history, resolved_intent, (dialogue or {}).get("referent"))
        docs = vault.db.execute(
            "SELECT id,filename,text,version,created_at,updated_at FROM documents "
            "WHERE patient_id=? AND state='approved' AND context_enabled=1 ORDER BY updated_at DESC",
            (patient_id,),
        ).fetchall()
        notes = vault.db.execute(
            "SELECT id,title,body,version,created_at,updated_at FROM notes "
            "WHERE patient_id=? AND context_enabled=1 ORDER BY updated_at DESC",
            (patient_id,),
        ).fetchall()
        candidates = []
        for kind, rows in [("R", docs), ("N", notes)]:
            for index, row in enumerate(rows, 1):
                raw_text = row["text"] if kind == "R" else row["body"]
                cleaned = sanitize_record_text(raw_text)
                if not cleaned:
                    continue
                report_date, document_type = source_metadata(raw_text)
                chunks = [cleaned[offset:offset+1800] for offset in range(0, len(cleaned), 1600)] or [cleaned]
                if resolved_intent == INTENT_LATEST_REPORT:
                    best = chunks[0]
                else:
                    best = max(chunks, key=lambda chunk: source_score(chunk, resolved_intent, terms))
                score = source_score(best, resolved_intent, terms)
                if kind == "R" and document_type:
                    display_title = document_type
                elif kind == "R":
                    display_title = Path(row["filename"]).stem.replace("-", " ").replace("_", " ").strip().title()
                else:
                    display_title = row["title"]
                record_date = report_date or row["updated_at"] or row["created_at"]
                candidates.append({
                    "label": f"{kind}{index}", "id": row["id"], "title": display_title,
                    "version": row["version"], "excerpt": best[:1800],
                    "partial": len(cleaned) > len(best), "score": score, "kind": kind,
                    "record_date": record_date, "document_type": document_type,
                    "_retrieval_text": cleaned,
                    "_retrieval_kind": (document_type or ("note" if kind == "N" else "document")).casefold().replace(" ", "_"),
                })

        fts_rank: dict[str, int] = {}
        if question.strip() and candidates:
            scoped_documents = [
                RetrievalDocument(
                    id=candidate["id"], patient_key=patient_id,
                    source_kind=candidate["_retrieval_kind"], title=candidate["title"],
                    text=candidate["_retrieval_text"],
                )
                for candidate in candidates
            ]
            for rank, hit in enumerate(
                FTS5Retriever().rank(question, scoped_documents, limit=len(scoped_documents)),
                1,
            ):
                fts_rank[hit.document_id] = rank
            for candidate in candidates:
                rank = fts_rank.get(candidate["id"])
                candidate["fts_rank"] = rank
                candidate["combined_score"] = candidate["score"] + (max(0, 12 - rank) if rank else 0)
        else:
            for candidate in candidates:
                candidate["fts_rank"] = None
                candidate["combined_score"] = candidate["score"]

        if not question.strip():
            # The context-preview endpoint is an inventory, not a retrieval
            # result. Return every unique active source so review/context tests
            # and the UI can see newly approved evidence immediately.
            ordered = sorted(candidates, key=lambda item: str(item.get("record_date") or ""), reverse=True)
            selected = deduplicate_sources(ordered, max(1, len(ordered)))
        elif resolved_intent == INTENT_LATEST_REPORT:
            document_candidates = [candidate for candidate in candidates if candidate["kind"] == "R"]
            pool = document_candidates or candidates
            ordered = sorted(pool, key=lambda item: str(item.get("record_date") or ""), reverse=True)
            selected = deduplicate_sources(ordered, 1)
        else:
            positives = [candidate for candidate in candidates if candidate["score"] > 0]
            pool = positives or candidates
            pool = sorted(
                pool,
                key=lambda item: (item["combined_score"], item["score"], str(item.get("record_date") or "")),
                reverse=True,
            )
            limit = 4 if resolved_intent == INTENT_OPEN else 3 if resolved_intent == INTENT_OVERVIEW else 2
            selected = deduplicate_sources(pool, limit)

        # Relabel after filtering so citations remain compact and do not expose gaps.
        counts = {"R": 0, "N": 0}
        for source in selected:
            counts[source["kind"]] += 1
            source["label"] = f"{source['kind']}{counts[source['kind']]}"
            source.pop("_retrieval_text", None)
            source.pop("_retrieval_kind", None)
        return {
            "patient_id": patient_id,
            "revision": patient["revision"],
            "sources": selected,
            "resolved_intent": resolved_intent,
            "retrieval_backend": "patient_scoped_fts5+intent_score" if question.strip() else "inventory",
            "total_active_sources": len(candidates),
            "retrieval_limited": len(candidates) > len(selected) or any(source["partial"] for source in selected),
        }

    @app.post("/api/patients/{patient_id}/assistant/thread")
    def assistant_thread_open(patient_id: str, body: S.AssistantThreadOpen, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            patient = patient_access(user, patient_id)
            if body.scope == "session":
                session_access(user, body.session_id, patient_id)
            row, created, context_reset = open_thread(
                db,
                user_id=user["id"],
                patient_id=patient_id,
                scope=body.scope,
                session_id=body.session_id,
                chart_revision=patient["revision"],
            )
            if created:
                vault.audit(user["id"], patient_id, "assistant.thread_created", row["id"], {
                    "scope": body.scope,
                    "session_id": body.session_id,
                    "chart_revision": patient["revision"],
                })
            elif context_reset:
                vault.audit(user["id"], patient_id, "assistant.thread_context_reset", row["id"], {
                    "scope": body.scope,
                    "session_id": body.session_id,
                    "chart_revision": patient["revision"],
                    "reason": "Patient chart revision changed; previous messages remain visible but are not used as follow-up context.",
                })
            return assistant_thread_response(db, row, patient["revision"], context_reset)

    @app.get("/api/patients/{patient_id}/assistant/threads")
    def assistant_threads(patient_id: str, user=Depends(authenticated)):
        clinician(user)
        with vault.mutex:
            patient = patient_access(user, patient_id)
            return list_threads(
                vault.db,
                user_id=user["id"],
                patient_id=patient_id,
                current_revision=patient["revision"],
            )

    @app.get("/api/assistant/threads/{thread_id}")
    def assistant_thread_detail(thread_id: str, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            row = get_owned_thread(db, thread_id=thread_id, user_id=user["id"])
            if row is None:
                abort(404, "Assistant conversation not found.")
            patient = patient_access(user, row["patient_id"])
            if row["session_id"]:
                session_access(user, row["session_id"], row["patient_id"])
            row, _, context_reset = open_thread(
                db,
                user_id=user["id"],
                patient_id=row["patient_id"],
                scope=row["scope"],
                session_id=row["session_id"],
                chart_revision=patient["revision"],
            )
            return assistant_thread_response(db, row, patient["revision"], context_reset)

    @app.post("/api/assistant/threads/{thread_id}/clear")
    def assistant_thread_clear(thread_id: str, body: S.AssistantThreadClear, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction() as db:
            row = get_owned_thread(db, thread_id=thread_id, user_id=user["id"])
            if row is None:
                abort(404, "Assistant conversation not found.")
            patient = patient_access(user, row["patient_id"])
            deleted = clear_thread(db, thread_id=thread_id, chart_revision=patient["revision"])
            vault.audit(user["id"], row["patient_id"], "assistant.thread_cleared", thread_id, {
                "scope": row["scope"],
                "session_id": row["session_id"],
                "messages_removed": deleted,
                "reason": body.reason,
            })
            refreshed = db.execute("SELECT * FROM assistant_threads WHERE id=?", (thread_id,)).fetchone()
            return assistant_thread_response(db, refreshed, patient["revision"], False)

    @app.get("/api/patients/{patient_id}/context")
    def context_preview(patient_id: str, user=Depends(authenticated)):
        clinician(user)
        with vault.transaction():
            context = context_for(user, patient_id)
            vault.audit(user["id"], patient_id, "context.previewed", patient_id, {"source_ids": [s["id"] for s in context["sources"]]})
            return context

    @app.post("/api/patients/{patient_id}/ask")
    def ask(patient_id: str, body: S.Question, request: Request, user=Depends(authenticated)):
        clinician(user)
        thread_id = body.thread_id
        thread_version = None
        thread_context_reset = False
        prior_dialogue_state = None
        workflow_result = None
        workflow_sources = None
        fact_answer = None
        fact_answer_facts = []
        fact_answer_sources = []
        with vault.transaction() as db:
            patient = patient_access(user, patient_id)
            if thread_id:
                thread = get_owned_thread(
                    db,
                    thread_id=thread_id,
                    user_id=user["id"],
                    patient_id=patient_id,
                )
                if thread is None:
                    abort(404, "Assistant conversation not found for this user and patient.")
                if thread["session_id"]:
                    session_access(user, thread["session_id"], patient_id)
                thread, _, thread_context_reset = open_thread(
                    db,
                    user_id=user["id"],
                    patient_id=patient_id,
                    scope=thread["scope"],
                    session_id=thread["session_id"],
                    chart_revision=patient["revision"],
                )
                thread_version = thread["version"]
                history = history_for_thread(
                    db,
                    thread_id=thread_id,
                    current_revision=patient["revision"],
                )
                prior_dialogue_state = get_dialogue_state(
                    db,
                    thread_id=thread_id,
                    current_revision=patient["revision"],
                )
            else:
                # Legacy clients may still supply bounded browser history. The
                # Stage 2 UI always uses an encrypted server-side thread.
                history = history_dicts(body.history)

            resolution = resolve_dialogue_state(body.question, prior_dialogue_state, history)
            context = context_for(
                user,
                patient_id,
                body.question,
                history,
                resolution.intent,
                resolution.public(),
            )
            resolution = enrich_dialogue_resolution(resolution, context["sources"], body.question)
            workflow_result = search_medication_event(
                db,
                patient_id=patient_id,
                question=body.question,
                resolution=resolution,
                history=history,
            )
            if workflow_result is not None and workflow_result.handled:
                workflow_sources = load_eligible_sources(db, patient_id)
                if workflow_result.clarification_kind:
                    apply_workflow_clarification(resolution, workflow_result)
                else:
                    stamp_event_resolution(resolution, workflow_result)
            elif resolution.relation in {"new_question", "confirmation", "format_change"}:
                fact_result = answer_from_facts(db, patient_id, resolution.intent, body.question)
                if fact_result is not None:
                    fact_answer, fact_answer_facts = fact_result
                    fact_answer_sources = fact_sources_for_ui(fact_answer_facts)
            if context["revision"] != body.expected_revision:
                abort(409, "Chart changed. Refresh before asking; old context is not reused.")
            generation = vault.generation
            question_hash = hashlib.sha256(body.question.encode("utf-8")).hexdigest()
            vault.audit(
                user["id"],
                patient_id,
                "assistant.requested",
                patient_id,
                {
                    "revision": context["revision"],
                    "source_ids": [source["id"] for source in context["sources"]],
                    "question_hash": question_hash,
                    "history_turns": len(history),
                    "resolved_intent": resolution.intent,
                    "dialogue_relation": resolution.relation,
                    "resolver_confidence": round(resolution.confidence, 4),
                    "clarification_required": resolution.clarification_required,
                    "referent_type": (resolution.referent or {}).get("type"),
                    "workflow_route": workflow_result.route if workflow_result else None,
                    "workflow_status": workflow_result.status if workflow_result else None,
                    "workflow_event_type": workflow_result.event_type if workflow_result else None,
                    "workflow_searched_sources": workflow_result.searched_source_count if workflow_result else 0,
                    "fact_store_answer": bool(fact_answer),
                    "fact_store_fact_count": len(fact_answer_facts),
                    "thread_id": thread_id,
                    "history_source": "encrypted_thread" if thread_id else "legacy_client",
                    "destination": "configured_private_model",
                    "remote": model.remote,
                },
            )

        answer_origin = "model"
        model_used = False
        unverified = False
        model_runtime = None
        safety_mode = resolution.safety_mode
        response_sources = context["sources"]
        workflow_public = workflow_result.public() if workflow_result is not None else None
        if workflow_result is not None and workflow_sources is not None:
            response_sources = event_sources_for_ui(workflow_result, workflow_sources)
        elif fact_answer is not None:
            response_sources = fact_answer_sources

        if resolution.clarification_required:
            answer = resolution.clarification_question or "Which chart item do you mean?"
            answer_origin = "clarification"
            model_used = False
            unverified = False
            notice = "The assistant needs one detail before it can answer from the chart."
        elif resolution.safety_mode == "treatment_boundary":
            answer = treatment_boundary_answer(body.question, history)
            answer_origin = "safety_boundary"
            response_sources = []
            notice = "This assistant summarizes reviewed chart information but does not choose treatment."
        elif workflow_result is not None and workflow_result.handled:
            answer = workflow_result.answer
            answer_origin = "workflow_search"
            model_used = False
            unverified = False
            notice = (
                "Medication-event search used the current encrypted provenance fact index and reviewed source coverage. "
                "Report and visit dates are not substituted for medication-event dates."
            )
        elif fact_answer is not None:
            answer = fact_answer
            answer_origin = "provenance_fact_store"
            model_used = False
            unverified = False
            notice = "Answer assembled from the encrypted provenance fact store; every displayed fact is linked to a current source version and exact quote."
        elif not context["sources"]:
            answer = "I don’t see any approved records or notes in this chart yet. Review an upload or add a note, then ask again."
            answer_origin = "no_evidence"
            response_sources = []
            notice = "No reviewed chart evidence was available."
        else:
            structured_answer = structured_chart_answer(
                body.question,
                resolution.intent,
                context["sources"],
                resolution.public(),
            )
            if structured_answer is not None:
                answer = structured_answer
                answer_origin = "structured_chart"
                notice = "Answer assembled from reviewed chart facts. Check the original records for clinical use."
            else:
                prompt = "\n\n".join(
                    f'[{source["label"]}] {source["title"]}\n{source["excerpt"]}'
                    for source in context["sources"]
                )
                try:
                    answer_method = getattr(model, "answer_with_history", None)
                    if callable(answer_method):
                        raw_answer = answer_method(body.question, prompt, history)
                    else:
                        raw_answer = model.answer(body.question, prompt)
                except ModelUnavailable:
                    abort(503, "Local model unavailable or the request could not fit its safe context budget. Your chart is saved; no remote fallback was used.")

                metrics_method = getattr(model, "last_request_metrics", None)
                model_runtime = metrics_method() if callable(metrics_method) else None
                answer = clean_model_answer(raw_answer)
                answer_origin = "model"
                if low_quality_answer(answer) or unsafe_treatment_language(answer) or answer_looks_like_source_dump(raw_answer):
                    answer = structured_chart_answer(
                        body.question,
                        INTENT_OVERVIEW,
                        context["sources"],
                        resolution.public(),
                    )
                    if answer is None:
                        answer = fallback_record_answer(body.question, context["sources"])
                    answer_origin = "record_fallback"
                model_used = answer_origin == "model"
                unverified = model_used
                notice = "AI-assisted chart answer. Verify the source records before relying on it."

        answer = apply_requested_format(answer, resolution.requested_format)
        # Explicit word limits remain supported for legacy clients and are
        # also represented in requested_format for persisted dialogue state.
        answer = truncate_words(answer, requested_word_limit(body.question) or resolution.requested_format.get("word_limit"))

        message_ids = None
        final_thread_version = thread_version
        final_dialogue_state = resolution.public()
        clarification_state = resolution_clarification(resolution)
        state_source_ids = [source["id"] for source in response_sources]
        with vault.mutex:
            if not vault.unlocked or generation != vault.generation:
                abort(409, "Vault changed while answering. Output discarded.")
            live_user = authenticated(request)
            with vault.transaction() as db:
                current = patient_access(live_user, patient_id)
                if current["revision"] != context["revision"]:
                    abort(409, "Chart changed while answering. Output discarded; ask again with the current chart.")
                if workflow_result is not None and workflow_result.handled:
                    current_workflow_sources = load_eligible_sources(db, patient_id)
                    if not validate_source_snapshot(workflow_result.source_snapshot or {}, current_workflow_sources):
                        abort(409, "Reviewed chart sources changed while searching. Output discarded; ask again.")
                if fact_answer_facts and not validate_fact_provenance(db, fact_answer_facts):
                    abort(409, "A reviewed source changed after fact retrieval. Output discarded; ask again.")
                if thread_id:
                    current_thread = get_owned_thread(
                        db,
                        thread_id=thread_id,
                        user_id=live_user["id"],
                        patient_id=patient_id,
                    )
                    if current_thread is None:
                        abort(409, "Conversation changed while answering. Output discarded.")
                    if current_thread["version"] != thread_version:
                        abort(409, "Conversation changed in another tab while answering. Output discarded; ask again.")
                    user_message_id, assistant_message_id, _ = append_exchange(
                        db,
                        thread_id=thread_id,
                        question=body.question,
                        answer=answer,
                        answer_origin=answer_origin,
                        source_ids=state_source_ids,
                        chart_revision=current["revision"],
                        intent=resolution.intent,
                        requested_word_limit=resolution.requested_format.get("word_limit"),
                        safety_mode=safety_mode,
                        referent=resolution.referent,
                        requested_format=resolution.requested_format,
                        resolver_confidence=resolution.confidence,
                        relation=resolution.relation,
                        clarification=clarification_state,
                        inherited=resolution.inherited,
                        resolution_reason=resolution.reason,
                    )
                    message_ids = {
                        "user": user_message_id,
                        "assistant": assistant_message_id,
                    }
                    final_thread_version = db.execute(
                        "SELECT version FROM assistant_threads WHERE id=?", (thread_id,)
                    ).fetchone()[0]
                    final_dialogue_state = get_dialogue_state(
                        db,
                        thread_id=thread_id,
                        current_revision=current["revision"],
                    )
                vault.audit(
                    user["id"],
                    patient_id,
                    "assistant.completed",
                    patient_id,
                    {
                        "revision": current["revision"],
                        "source_ids": state_source_ids,
                        "model": model.model if model_used else None,
                        "answer_origin": answer_origin,
                        "resolved_intent": resolution.intent,
                        "dialogue_relation": resolution.relation,
                        "resolver_confidence": round(resolution.confidence, 4),
                        "clarification_required": resolution.clarification_required,
                        "referent_type": (resolution.referent or {}).get("type"),
                        "workflow_route": workflow_result.route if workflow_result else None,
                        "workflow_status": workflow_result.status if workflow_result else None,
                        "workflow_event_type": workflow_result.event_type if workflow_result else None,
                        "workflow_searched_sources": workflow_result.searched_source_count if workflow_result else 0,
                        "history_turns": len(history),
                        "thread_id": thread_id,
                        "message_ids": message_ids,
                        "model_runtime": model_runtime,
                        "answer_persisted": bool(thread_id),
                    },
                )

        visible_sources = [] if answer_origin == "clarification" else response_sources
        return {
            **context,
            "sources": visible_sources,
            "answer": answer,
            "model_used": model_used,
            "unverified": unverified,
            "answer_origin": answer_origin,
            "history_used": bool(history),
            "model_runtime": model_runtime,
            "notice": notice,
            "thread_id": thread_id,
            "thread_version": final_thread_version,
            "thread_context_reset": thread_context_reset,
            "messages_persisted": bool(thread_id),
            "message_ids": message_ids,
            "dialogue_state": final_dialogue_state,
            "clarification_required": resolution.clarification_required,
            "resolver_confidence": round(resolution.confidence, 4),
            "workflow_search": workflow_public,
        }

    class PatientArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        patient_id: str = Field(min_length=1, max_length=64)

    tool_definitions = [
        {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type":"object","properties":{"patient_id":{"type":"string"}},"required":["patient_id"],"additionalProperties":False}}}
        for name,description in [
            ("get_patient_context", "Read approved, included evidence for one authorized patient. No SQL or web access."),
            ("list_patient_documents", "List source metadata for one authorized patient; no network requests or writes."),
        ]
    ]

    @app.get("/api/tools")
    def tools(user=Depends(authenticated)):
        clinician(user)
        return {"tools": tool_definitions, "execution": "Application-controlled, authenticated, read-only.",
                "note": "OpenAI function schemas are not arbitrary connector APIs. The backend executes permitted calls."}

    @app.post("/api/tools/invoke")
    def invoke(body: S.ToolInvocation, user=Depends(authenticated)):
        clinician(user)
        try:
            args = PatientArgs.model_validate(body.arguments)
        except ValidationError:
            abort(422, "Tool requires exactly one patient_id and no extra fields.")
        with vault.transaction() as db:
            patient_access(user, args.patient_id)
            if body.name == "get_patient_context":
                result = context_for(user, args.patient_id)
            else:
                result = [document_public(r) for r in db.execute("SELECT * FROM documents WHERE patient_id=? AND state='approved' AND context_enabled=1", (args.patient_id,))]
            vault.audit(user["id"], args.patient_id, "tool.invoked", body.name, {})
            return {"name": body.name, "result": result}

    @app.get("/api/backup")
    def encrypted_backup(user=Depends(authenticated)):
        admin(user)
        with vault.transaction():
            vault.audit(user["id"], None, "vault.backup_exported", "vault", {})
        return Response(vault.path.read_bytes(), media_type="application/octet-stream",
                        headers={"Content-Disposition": 'attachment; filename="workspace-backup.vault"'})

    @app.post("/api/seed", status_code=201)
    def synthetic_seed(user=Depends(authenticated)):
        admin(user)
        with vault.transaction() as db:
            if db.execute("SELECT COUNT(*) FROM patients").fetchone()[0]:
                abort(409, "Sample data can only be loaded into an empty workspace.")
            people = DEMO_RECORDS
            for fixture in people:
                name, dob, mrn = fixture["name"], fixture["dob"], fixture["mrn"]
                body = S.PatientCreate(name=name, dob=dob, mrn=mrn)
                pid = insert_patient(body, user)
                note_id = uid(); timestamp = now()
                db.execute("INSERT INTO notes(id,patient_id,title,body,created_by,created_at,updated_at) VALUES(?,?,?,?,?,?,?)", (note_id,pid,fixture["title"],fixture["note"],user["id"],timestamp,timestamp))
                db.execute("INSERT INTO note_versions VALUES(?,?,?,?,?,?,?)", (note_id,1,fixture["title"],fixture["note"],user["id"],"Demo seed",timestamp))
                sync_source_facts(db, "note", note_id)
                vault.audit(user["id"],pid,"note.created",note_id,{"synthetic":True,"title":fixture["title"],"body":fixture["note"]})
                raw = fixture["source"].encode(); doc_id = uid()
                db.execute("""INSERT INTO documents(id,patient_id,filename,mime,original,sha256,text,summary,summary_kind,method,identity_json,state,context_enabled,created_by,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (doc_id,pid,fixture["filename"],"text/plain",raw,hashlib.sha256(raw).hexdigest(),fixture["source"],
                    fixture["summary"],"demo_seed","text",canonical(identity(fixture["source"])),"approved",1,user["id"],timestamp,timestamp))
                sync_source_facts(db, "document", doc_id)
                vault.audit(user["id"],pid,"document.seeded",doc_id,{"synthetic":True,"demo_record":True})
                bump(pid)
        return {"loaded": 3, "synthetic_only": True}

    static = ROOT / "app" / "static"
    if static.exists():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(static / "index.html")
    return app


app = create_app()
