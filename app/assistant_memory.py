"""Encrypted per-user assistant conversation memory.

Conversation history improves follow-up interpretation but is never patient evidence.
Reviewed chart evidence is always reloaded by the existing retrieval path.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from .vault import now, uid

MAX_VISIBLE_MESSAGES = 100
MAX_HISTORY_TURNS = 8
DEFAULT_TITLE = "Chart conversation"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _loads(value: str | None, default):
    try:
        return json.loads(value or "")
    except (TypeError, json.JSONDecodeError):
        return default


def dialogue_state_public(row: sqlite3.Row | None, *, current_revision: int | None = None) -> dict:
    if row is None:
        return {
            "active_intent": None,
            "referent": {},
            "requested_format": {},
            "safety_mode": "chart_only",
            "last_source_ids": [],
            "chart_revision": current_revision,
            "resolver_confidence": 0.0,
            "relation": None,
            "clarification": {},
            "last_resolution": {},
        }
    requested = _loads(row["requested_format_json"], {})
    resolver = requested.pop("resolver", {}) if isinstance(requested, dict) else {}
    clarification = {
        "required": bool(resolver.get("clarification_required")),
        "question": resolver.get("clarification_question"),
        "intent": resolver.get("intent"),
        "pending_relation": resolver.get("pending_relation"),
        "options": resolver.get("options") or [],
    }
    return {
        "active_intent": row["active_intent"],
        "referent": _loads(row["referent_json"], {}),
        "requested_format": requested if isinstance(requested, dict) else {},
        "safety_mode": row["safety_mode"],
        "last_source_ids": _loads(row["last_source_ids_json"], []),
        "chart_revision": row["chart_revision"],
        "resolver_confidence": float(resolver.get("confidence") or 0.0),
        "relation": resolver.get("relation"),
        "clarification": clarification,
        "last_resolution": resolver if isinstance(resolver, dict) else {},
    }


def get_dialogue_state(db: sqlite3.Connection, *, thread_id: str, current_revision: int) -> dict:
    row = db.execute(
        "SELECT * FROM assistant_dialogue_state WHERE thread_id=?", (thread_id,)
    ).fetchone()
    return dialogue_state_public(row, current_revision=current_revision)


def thread_public(row: sqlite3.Row, *, message_count: int = 0, context_reset: bool = False) -> dict:
    return {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "session_id": row["session_id"],
        "scope": row["scope"],
        "title": row["title"],
        "status": row["status"],
        "chart_revision": row["chart_revision"],
        "version": row["version"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "message_count": message_count,
        "context_reset": context_reset,
    }


def message_public(row: sqlite3.Row, *, current_revision: int) -> dict:
    return {
        "id": row["id"],
        "sequence": row["sequence"],
        "role": row["role"],
        "content": row["content"],
        "answer_origin": row["answer_origin"],
        "source_ids": _loads(row["source_ids_json"], []),
        "chart_revision": row["chart_revision"],
        "created_at": row["created_at"],
        "stale": row["chart_revision"] != current_revision,
    }


def _thread_query(scope: str) -> str:
    if scope == "longitudinal":
        return (
            "SELECT * FROM assistant_threads WHERE user_id=? AND patient_id=? "
            "AND scope='longitudinal' AND session_id IS NULL AND status='active'"
        )
    return (
        "SELECT * FROM assistant_threads WHERE user_id=? AND patient_id=? "
        "AND scope='session' AND session_id=? AND status='active'"
    )


def find_active_thread(
    db: sqlite3.Connection,
    *,
    user_id: str,
    patient_id: str,
    scope: str,
    session_id: str | None,
) -> sqlite3.Row | None:
    if scope == "longitudinal":
        return db.execute(_thread_query(scope), (user_id, patient_id)).fetchone()
    return db.execute(_thread_query(scope), (user_id, patient_id, session_id)).fetchone()


def _reset_dialogue_state(db: sqlite3.Connection, thread_id: str, chart_revision: int, timestamp: str) -> None:
    db.execute(
        """INSERT INTO assistant_dialogue_state(
             thread_id,active_intent,referent_json,requested_format_json,safety_mode,
             last_source_ids_json,chart_revision,updated_at
           ) VALUES(?,NULL,'{}','{}','chart_only','[]',?,?)
           ON CONFLICT(thread_id) DO UPDATE SET
             active_intent=NULL,referent_json='{}',requested_format_json='{}',
             safety_mode='chart_only',last_source_ids_json='[]',
             chart_revision=excluded.chart_revision,updated_at=excluded.updated_at""",
        (thread_id, chart_revision, timestamp),
    )


def open_thread(
    db: sqlite3.Connection,
    *,
    user_id: str,
    patient_id: str,
    scope: str,
    session_id: str | None,
    chart_revision: int,
) -> tuple[sqlite3.Row, bool, bool]:
    """Return active thread, whether it was created, and whether state was reset."""
    if scope not in {"longitudinal", "session"}:
        raise ValueError("Invalid assistant thread scope.")
    if scope == "longitudinal" and session_id is not None:
        raise ValueError("Longitudinal thread cannot have a session_id.")
    if scope == "session" and not session_id:
        raise ValueError("Session thread requires session_id.")

    row = find_active_thread(
        db,
        user_id=user_id,
        patient_id=patient_id,
        scope=scope,
        session_id=session_id,
    )
    timestamp = now()
    created = False
    context_reset = False
    if row is None:
        thread_id = uid()
        db.execute(
            """INSERT INTO assistant_threads(
                 id,user_id,patient_id,session_id,scope,title,status,chart_revision,
                 version,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,'active',?,1,?,?)""",
            (
                thread_id, user_id, patient_id, session_id, scope, DEFAULT_TITLE,
                chart_revision, timestamp, timestamp,
            ),
        )
        _reset_dialogue_state(db, thread_id, chart_revision, timestamp)
        row = db.execute("SELECT * FROM assistant_threads WHERE id=?", (thread_id,)).fetchone()
        created = True
    elif row["chart_revision"] != chart_revision:
        db.execute(
            "UPDATE assistant_threads SET chart_revision=?,version=version+1,updated_at=? WHERE id=?",
            (chart_revision, timestamp, row["id"]),
        )
        _reset_dialogue_state(db, row["id"], chart_revision, timestamp)
        row = db.execute("SELECT * FROM assistant_threads WHERE id=?", (row["id"],)).fetchone()
        context_reset = True
    return row, created, context_reset


def get_owned_thread(
    db: sqlite3.Connection,
    *,
    thread_id: str,
    user_id: str,
    patient_id: str | None = None,
) -> sqlite3.Row | None:
    row = db.execute(
        "SELECT * FROM assistant_threads WHERE id=? AND user_id=? AND status='active'",
        (thread_id, user_id),
    ).fetchone()
    if row is None or (patient_id is not None and row["patient_id"] != patient_id):
        return None
    return row


def list_messages(
    db: sqlite3.Connection,
    *,
    thread_id: str,
    current_revision: int,
    limit: int = MAX_VISIBLE_MESSAGES,
) -> tuple[list[dict], bool]:
    count = db.execute(
        "SELECT COUNT(*) FROM assistant_messages WHERE thread_id=?", (thread_id,)
    ).fetchone()[0]
    rows = db.execute(
        """SELECT * FROM (
             SELECT * FROM assistant_messages WHERE thread_id=?
             ORDER BY sequence DESC LIMIT ?
           ) ORDER BY sequence""",
        (thread_id, limit),
    ).fetchall()
    return [message_public(row, current_revision=current_revision) for row in rows], count > limit


def history_for_thread(
    db: sqlite3.Connection,
    *,
    thread_id: str,
    current_revision: int,
    limit: int = MAX_HISTORY_TURNS,
) -> list[dict]:
    rows = db.execute(
        """SELECT role,content FROM (
             SELECT role,content,sequence FROM assistant_messages
             WHERE thread_id=? AND chart_revision=?
             ORDER BY sequence DESC LIMIT ?
           ) ORDER BY sequence""",
        (thread_id, current_revision, limit),
    ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def list_threads(
    db: sqlite3.Connection,
    *,
    user_id: str,
    patient_id: str,
    current_revision: int,
) -> list[dict]:
    rows = db.execute(
        """SELECT t.*,COUNT(m.id) AS message_count
           FROM assistant_threads t
           LEFT JOIN assistant_messages m ON m.thread_id=t.id
           WHERE t.user_id=? AND t.patient_id=? AND t.status='active'
           GROUP BY t.id ORDER BY t.updated_at DESC""",
        (user_id, patient_id),
    ).fetchall()
    return [
        thread_public(
            row,
            message_count=row["message_count"],
            context_reset=row["chart_revision"] != current_revision,
        )
        for row in rows
    ]


def append_exchange(
    db: sqlite3.Connection,
    *,
    thread_id: str,
    question: str,
    answer: str,
    answer_origin: str,
    source_ids: list[str],
    chart_revision: int,
    intent: str,
    requested_word_limit: int | None,
    safety_mode: str,
    referent: dict | None = None,
    requested_format: dict | None = None,
    resolver_confidence: float = 0.0,
    relation: str | None = None,
    clarification: dict | None = None,
    inherited: bool = False,
    resolution_reason: str | None = None,
) -> tuple[str, str, int]:
    row = db.execute("SELECT * FROM assistant_threads WHERE id=?", (thread_id,)).fetchone()
    if row is None:
        raise ValueError("Assistant thread no longer exists.")
    timestamp = now()
    next_sequence = db.execute(
        "SELECT COALESCE(MAX(sequence),0)+1 FROM assistant_messages WHERE thread_id=?",
        (thread_id,),
    ).fetchone()[0]
    user_message_id = uid()
    assistant_message_id = uid()
    db.execute(
        """INSERT INTO assistant_messages(
             id,thread_id,sequence,role,content,answer_origin,source_ids_json,
             chart_revision,created_at
           ) VALUES(?,?,?,'user',?,NULL,'[]',?,?)""",
        (user_message_id, thread_id, next_sequence, question, chart_revision, timestamp),
    )
    db.execute(
        """INSERT INTO assistant_messages(
             id,thread_id,sequence,role,content,answer_origin,source_ids_json,
             chart_revision,created_at
           ) VALUES(?,?,?,'assistant',?,?,?,?,?)""",
        (
            assistant_message_id, thread_id, next_sequence + 1, answer, answer_origin,
            _json(source_ids), chart_revision, timestamp,
        ),
    )
    title = row["title"]
    if title == DEFAULT_TITLE:
        compact = " ".join(question.split())
        title = compact[:77] + ("…" if len(compact) > 77 else "")
    db.execute(
        """UPDATE assistant_threads SET title=?,chart_revision=?,version=version+1,
           updated_at=? WHERE id=?""",
        (title, chart_revision, timestamp, thread_id),
    )
    db.execute(
        """INSERT INTO assistant_dialogue_state(
             thread_id,active_intent,referent_json,requested_format_json,safety_mode,
             last_source_ids_json,chart_revision,updated_at
           ) VALUES(?,?,?,?,?,?,?,?)
           ON CONFLICT(thread_id) DO UPDATE SET
             active_intent=excluded.active_intent,
             referent_json=excluded.referent_json,
             requested_format_json=excluded.requested_format_json,
             safety_mode=excluded.safety_mode,
             last_source_ids_json=excluded.last_source_ids_json,
             chart_revision=excluded.chart_revision,updated_at=excluded.updated_at""",
        (
            thread_id,
            intent,
            _json(referent or {}),
            _json({
                **(requested_format or ({"word_limit": requested_word_limit} if requested_word_limit else {})),
                "resolver": {
                    "confidence": round(float(resolver_confidence), 4),
                    "relation": relation,
                    "clarification_required": bool((clarification or {}).get("required")),
                    "clarification_question": (clarification or {}).get("question"),
                    "inherited": bool(inherited),
                    "reason": resolution_reason,
                    "options": (clarification or {}).get("options", []),
                    "pending_relation": (clarification or {}).get("pending_relation"),
                    "intent": (clarification or {}).get("intent"),
                },
            }),
            safety_mode,
            _json(source_ids),
            chart_revision,
            timestamp,
        ),
    )
    return user_message_id, assistant_message_id, next_sequence + 1


def clear_thread(db: sqlite3.Connection, *, thread_id: str, chart_revision: int) -> int:
    deleted = db.execute(
        "SELECT COUNT(*) FROM assistant_messages WHERE thread_id=?", (thread_id,)
    ).fetchone()[0]
    db.execute("DELETE FROM assistant_messages WHERE thread_id=?", (thread_id,))
    timestamp = now()
    db.execute(
        """UPDATE assistant_threads SET title=?,chart_revision=?,version=version+1,
           updated_at=? WHERE id=?""",
        (DEFAULT_TITLE, chart_revision, timestamp, thread_id),
    )
    _reset_dialogue_state(db, thread_id, chart_revision, timestamp)
    return deleted
