"""Patient-scoped medication-event retrieval for the chart assistant.

This module handles a narrow but important class of questions that should not
fall back to the generic current-medication template:

    when was lisinopril prescribed?
    when was lisinopril dispensed?
    when did the patient pick it up?
    when was it started?

The workflow scans only already-authorized, approved/context-enabled sources for
one patient. It uses an ephemeral SQLite FTS5 index for relevance ranking, then
validates explicit medication -> event -> date statements against the full
eligible source set. Report/visit dates are never substituted for medication
event dates.

It is intentionally conservative. Ambiguous, negated, multi-drug, or
multi-date statements are surfaced for review rather than converted into a
confident date claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import re
import sqlite3
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .chart_assistant import INTENT_MEDICATIONS

MAX_SOURCES = 256
MAX_SOURCE_CHARACTERS = 1_500_000
MAX_MEDICATION_CANDIDATES = 40
MAX_CITATIONS = 12

EventType = Literal["prescribed", "renewed", "dispensed", "picked_up", "started", "discontinued", "restarted"]


class MedicationEventTask(BaseModel):
    """Validated task contract; authority/patient scope is supplied by the app."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    medications: tuple[str, ...] = Field(min_length=1, max_length=8)
    event_type: EventType
    requested_field: Literal["date"] = "date"
    wording: Literal["explicit", "requested_alias", "clarification_selection"] = "explicit"

    @model_validator(mode="after")
    def normalized_medications(self):
        unique = tuple(dict.fromkeys(name.casefold().strip() for name in self.medications if name.strip()))
        if not unique:
            raise ValueError("Medication-event task requires at least one medication.")
        object.__setattr__(self, "medications", unique)
        return self


@dataclass(frozen=True)
class EligibleSource:
    id: str
    kind: str
    title: str
    body: str
    version: int
    record_date: str | None
    updated_at: str | None

    def snapshot(self) -> tuple[int, str]:
        return self.version, hashlib.sha256(self.body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MedicationEventCitation:
    source_id: str
    source_version: int
    source_title: str
    start: int
    end: int
    quote: str
    medication: str
    event_type: str
    event_date: str
    event_role: str = "unspecified"


@dataclass
class MedicationEventResult:
    handled: bool
    route: str = "medication_event"
    status: str = "not_applicable"
    answer: str = ""
    coverage: str = "not_applicable"
    citations: list[MedicationEventCitation] | None = None
    medications: tuple[str, ...] = ()
    event_type: str | None = None
    eligible_source_count: int = 0
    searched_source_count: int = 0
    clarification_kind: str | None = None
    clarification_question: str | None = None
    clarification_options: list[dict[str, Any]] | None = None
    source_snapshot: dict[str, tuple[int, str]] | None = None
    ranked_source_ids: list[str] | None = None

    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.clarification_options is None:
            self.clarification_options = []
        if self.source_snapshot is None:
            self.source_snapshot = {}
        if self.ranked_source_ids is None:
            self.ranked_source_ids = []

    @property
    def source_ids(self) -> list[str]:
        ids: list[str] = []
        for citation in self.citations or []:
            if citation.source_id not in ids:
                ids.append(citation.source_id)
        return ids

    def public(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "status": self.status,
            "coverage": self.coverage,
            "event_type": self.event_type,
            "medications": list(self.medications),
            "eligible_source_count": self.eligible_source_count,
            "searched_source_count": self.searched_source_count,
            "clarification_kind": self.clarification_kind,
            "ranked_source_ids": list(self.ranked_source_ids or []),
            "citations": [
                {
                    "source_id": c.source_id,
                    "source_version": c.source_version,
                    "source_title": c.source_title,
                    "start": c.start,
                    "end": c.end,
                    "quote": c.quote,
                    "medication": c.medication,
                    "event_type": c.event_type,
                    "event_date": c.event_date,
                    "event_role": c.event_role,
                }
                for c in (self.citations or [])
            ],
        }


_DATE_PATTERN = r"(?:\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})"
_DRUG_WITH_DOSE = re.compile(
    r"(?i)(?:^|[;\n])\s*(?:current\s+medications?\s*:\s*|current\s+medication\s*:\s*|medications?\s*:\s*)?"
    r"\s*([a-z][a-z-]{2,35})(?:\s+(?:\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|units?)\b))"
)
_ORDERED_DRUG = re.compile(
    r"(?i)\b([a-z][a-z-]{2,35})"
    r"(?:\s+(?:prescription|order|rx)\s*(?:id|number|#)\s*[:#-]?\s*[A-Za-z0-9._-]{2,40})?"
    r"\s+(?:was\s+)?(?:(?:newly|first|initially|originally)\s+)?"
    r"(?:(?:not|never|possibly|maybe)\s+)?(?:prescribed|ordered|renewed|refilled|reordered|dispensed|filled|started|began|discontinued|stopped|restarted|resumed)\b"
)
_EVENT_PATTERNS: dict[str, re.Pattern[str]] = {
    "prescribed": re.compile(r"(?i)\bprescrib(?:ed|ing|e)|\bprescription(?:\s+date)?\b|\border(?:ed|ing)?\b"),
    "renewed": re.compile(r"(?i)\brenew(?:ed|al)?\b|\brefill(?:ed)?\b|\breorder(?:ed)?\b"),
    "dispensed": re.compile(r"(?i)\bdispens(?:ed|ing)|\bfill(?:ed)?\b"),
    "picked_up": re.compile(r"(?i)\bpick(?:ed)?[ -]?up\b|\bcollected\b"),
    "started": re.compile(r"(?i)\b(?:started|began)\b"),
    "discontinued": re.compile(r"(?i)\b(?:discontinued|stopped|ceased)\b"),
    "restarted": re.compile(r"(?i)\b(?:restarted|resumed)\b"),
}
_EVENT_VERB: dict[str, str] = {
    "prescribed": r"(?:prescrib(?:ed|ing)|ordered|prescription\s+dated?)",
    "renewed": r"(?:renewed|refilled|reordered)",
    "dispensed": r"(?:dispens(?:ed|ing)|filled)",
    "picked_up": r"(?:picked[ -]?up|collected)",
    "started": r"(?:started|began)",
    "discontinued": r"(?:discontinued|stopped|ceased)",
    "restarted": r"(?:restarted|resumed)",
}
_NEGATIVE_OR_UNCERTAIN = re.compile(
    r"(?i)\b(?:not|never|denied|no|without|unconfirmed|uncertain|possibly|might|maybe|reportedly)\b"
)
_DATE_QUESTION = re.compile(r"(?i)\b(?:when|what\s+date|which\s+date|date\s+of|on\s+what\s+date)\b")
_RECEIVED_WORDING = re.compile(r"(?i)\b(?:received|receive|got|obtained)\b")
_REQUESTED_WORDING = re.compile(r"(?i)\brequest(?:ed|ing)?\b")
_BOTH_WORDING = re.compile(r"(?i)^\s*(?:both|all|all\s+of\s+them|both\s+of\s+them)\s*[.?!]*$")


def _normalize_date(raw: str) -> str | None:
    value = raw.strip()
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        try:
            return datetime.strptime(value.replace(",", ""), "%B %d %Y").date().isoformat()
        except ValueError:
            return None


def load_eligible_sources(db: sqlite3.Connection, patient_id: str) -> list[EligibleSource]:
    """Load the longitudinal reviewed source set for one already-authorized patient."""
    documents = db.execute(
        """SELECT id,filename,text,version,updated_at FROM documents
           WHERE patient_id=? AND state='approved' AND context_enabled=1
           ORDER BY updated_at DESC""",
        (patient_id,),
    ).fetchall()
    notes = db.execute(
        """SELECT id,title,body,version,updated_at FROM notes
           WHERE patient_id=? AND context_enabled=1
           ORDER BY updated_at DESC""",
        (patient_id,),
    ).fetchall()
    output: list[EligibleSource] = []
    report_date_re = re.compile(r"(?im)^\s*(?:report|document)\s+date\s*:\s*(\d{4}-\d{2}-\d{2})\s*$")
    visit_date_re = re.compile(r"(?im)\b(?:visit|review)\s*\(?\s*(\d{4}-\d{2}-\d{2})\s*\)?")
    for row in documents:
        body = str(row["text"] or "")
        match = report_date_re.search(body)
        output.append(EligibleSource(
            id=row["id"], kind="R", title=row["filename"], body=body,
            version=int(row["version"]), record_date=match.group(1) if match else None,
            updated_at=row["updated_at"],
        ))
    for row in notes:
        body = str(row["body"] or "")
        match = visit_date_re.search(body)
        output.append(EligibleSource(
            id=row["id"], kind="N", title=row["title"], body=body,
            version=int(row["version"]), record_date=match.group(1) if match else None,
            updated_at=row["updated_at"],
        ))
    return output


def source_snapshot(sources: list[EligibleSource]) -> dict[str, tuple[int, str]]:
    return {source.id: source.snapshot() for source in sources}


def _known_medications(sources: list[EligibleSource]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for line in source.body.splitlines():
            lower = line.casefold()
            if not any(word in lower for word in ("medication", "prescrib", "ordered", "renew", "refill", "reorder", "dispens", "filled", "started", "discontinu", "stopped", "restart", "resum")):
                continue
            for item in re.split(r"[;,]|\s+and\s+", line):
                candidates: list[str] = []
                found = _DRUG_WITH_DOSE.search(item)
                if found:
                    candidates.append(found.group(1).casefold())
                ordered = _ORDERED_DRUG.search(item)
                if ordered:
                    candidates.append(ordered.group(1).casefold())
                for name in candidates:
                    if name not in seen:
                        seen.add(name)
                        names.append(name)
    return tuple(names)


def _event_type(question: str) -> str | None:
    found = [event for event, pattern in _EVENT_PATTERNS.items() if pattern.search(question)]
    if len(found) == 1:
        return found[0]
    return None


def _dialogue_items(resolution: Any) -> list[dict[str, Any]]:
    referent = getattr(resolution, "referent", {}) or {}
    selected = referent.get("selected") if isinstance(referent, dict) else None
    if isinstance(selected, dict) and selected.get("selection") == "all":
        return [item for item in selected.get("items", []) if isinstance(item, dict)]
    if isinstance(selected, dict) and selected.get("label"):
        return [selected]
    return [item for item in referent.get("items", []) if isinstance(item, dict)] if isinstance(referent, dict) else []


def _selected_medications(
    question: str,
    known: tuple[str, ...],
    resolution: Any,
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, ...]:
    explicit = tuple(name for name in known if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question, re.I))
    if explicit:
        return explicit
    referent = getattr(resolution, "referent", {}) or {}
    stored_selected = tuple(
        name for name in (referent.get("selected_medications") or [])
        if isinstance(name, str) and name.casefold() in known
    ) if isinstance(referent, dict) else ()
    if stored_selected:
        return tuple(name.casefold() for name in stored_selected)
    medication_items = referent.get("medication_items") or _dialogue_items(resolution)
    selected = referent.get("selected") if isinstance(referent, dict) else None
    if isinstance(selected, dict) and selected.get("selection") == "all":
        labels = [str(item.get("label") or "").casefold() for item in selected.get("items", [])]
        ordered = [label for label in labels if label in known]
        return tuple(dict.fromkeys(ordered))
    if isinstance(selected, dict):
        label = str(selected.get("label") or selected.get("value") or "").casefold()
        hits = tuple(name for name in known if name == label or re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", label, re.I))
        if hits:
            return hits
    if _BOTH_WORDING.match(question):
        labels = [str(item.get("label") or "").casefold() for item in medication_items if isinstance(item, dict)]
        hits = tuple(name for name in known if not labels or name in labels)
        return hits or known
    for turn in reversed((history or [])[-6:]):
        if str(turn.get("role") or "") != "user":
            continue
        text = str(turn.get("content") or "")
        hits = tuple(name for name in known if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I))
        if hits:
            return hits
    return ()


def _explicit_event_from_resolution(question: str, resolution: Any) -> tuple[str | None, str]:
    event = _event_type(question)
    if event:
        return event, "explicit"
    relation = str(getattr(resolution, "relation", "") or "")
    referent = getattr(resolution, "referent", {}) or {}
    selected = referent.get("selected") if isinstance(referent, dict) else None
    if relation == "medication_event_date" and isinstance(selected, dict) and selected.get("event_type"):
        return str(selected["event_type"]), "clarification_selection"
    if relation in {"prescribed_date", "request_date"} and getattr(resolution, "intent", None) == INTENT_MEDICATIONS:
        return "prescribed", "requested_alias" if relation == "request_date" else "explicit"
    if relation == "dispensed_date" and getattr(resolution, "intent", None) == INTENT_MEDICATIONS:
        return "dispensed", "explicit"
    if relation == "pickup_date" and getattr(resolution, "intent", None) == INTENT_MEDICATIONS:
        return "picked_up", "explicit"
    if relation == "started_date" and getattr(resolution, "intent", None) == INTENT_MEDICATIONS:
        return "started", "explicit"
    return None, "explicit"


def _is_medication_event_query(question: str, resolution: Any, known: tuple[str, ...]) -> bool:
    intent = getattr(resolution, "intent", None)
    explicit_event = _event_type(question)
    explicit_medication = any(
        re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question, re.I)
        for name in known
    )
    if intent != INTENT_MEDICATIONS and not (
        _DATE_QUESTION.search(question)
        and explicit_medication
        and (explicit_event or _RECEIVED_WORDING.search(question) or _REQUESTED_WORDING.search(question))
    ):
        return False
    relation = str(getattr(resolution, "relation", "") or "")
    if relation in {"request_date", "received_status", "prescribed_date", "dispensed_date", "pickup_date", "started_date", "medication_event_date"}:
        return True
    if _DATE_QUESTION.search(question) and (
        _event_type(question)
        or _REQUESTED_WORDING.search(question)
        or _RECEIVED_WORDING.search(question)
        or any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", question, re.I) for name in known)
    ):
        return True
    return False


def _rank_sources_fts(sources: list[EligibleSource], question: str, limit: int = 4) -> list[str]:
    tokens = list(dict.fromkeys(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,35}", question.casefold())))[:20]
    if not tokens or not sources:
        return []
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("CREATE VIRTUAL TABLE scoped_fts USING fts5(source_id UNINDEXED, body)")
        conn.executemany("INSERT INTO scoped_fts(source_id,body) VALUES(?,?)", [(s.id, s.body) for s in sources])
        match = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        return [
            row[0]
            for row in conn.execute(
                "SELECT source_id FROM scoped_fts WHERE scoped_fts MATCH ? ORDER BY bm25(scoped_fts) LIMIT ?",
                (match, min(8, max(1, limit))),
            )
        ]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _event_mentions(
    source: EligibleSource,
    medication: str,
    event_type: str,
    known_medications: tuple[str, ...],
) -> tuple[list[MedicationEventCitation], bool]:
    verb = _EVENT_VERB[event_type]
    results: list[MedicationEventCitation] = []
    ambiguous = False
    for segment in re.finditer(r"[^\n.!?;]+(?:[.!?;]|$)", source.body):
        excerpt = segment.group(0).strip()
        if not excerpt:
            continue
        medication_pattern = r"(?<!\w)" + re.escape(medication) + r"(?!\w)"
        if not re.search(medication_pattern, excerpt, re.I):
            continue
        if not re.search(r"(?i)\b" + verb + r"\b", excerpt):
            continue
        named = [
            name for name in known_medications
            if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", excerpt, re.I)
        ]
        if len(named) > 1:
            ambiguous = True
            continue
        relationship = (
            medication_pattern
            + r"(?:\s+(?:prescription|order|rx)\s*(?:id|number|#)\s*[:#-]?\s*[A-Za-z0-9._-]{2,40})?"
            + r"\s+(?:was\s+|is\s+|has\s+been\s+)?(?:(?:newly|first|originally)\s+)?"
            + verb
            + r"\s+(?:(?:on|dated|date)\s+)?"
            + _DATE_PATTERN
        )
        reverse = (
            r"(?i)\b(?:prescription|dispensing|pickup|start)\s+(?:for|of)\s+"
            + medication_pattern + r"\s+(?:was\s+)?(?:dated|on)\s+" + _DATE_PATTERN
        )
        other_event = any(
            other != event_type and re.search(r"(?i)\b" + other_verb + r"\b", excerpt)
            for other, other_verb in _EVENT_VERB.items()
        )
        if other_event or not (re.search(relationship, excerpt, re.I) or (event_type == "prescribed" and re.search(reverse, excerpt, re.I))):
            ambiguous = True
            continue
        dates = [_normalize_date(match.group()) for match in re.finditer(_DATE_PATTERN, excerpt)]
        dates = [value for value in dates if value]
        if _NEGATIVE_OR_UNCERTAIN.search(excerpt) or len(set(dates)) != 1:
            ambiguous = True
            continue
        leading = len(segment.group(0)) - len(segment.group(0).lstrip())
        start = segment.start() + leading
        end = start + len(excerpt)
        if source.body[start:end] != excerpt:
            ambiguous = True
            continue
        results.append(MedicationEventCitation(
            source_id=source.id,
            source_version=source.version,
            source_title=source.title,
            start=start,
            end=end,
            quote=excerpt,
            medication=medication,
            event_type=event_type,
            event_date=dates[0],
            event_role=(
                "renewal" if event_type == "renewed" else
                "discontinuation" if event_type == "discontinued" else
                "restart" if event_type == "restarted" else
                "original" if event_type == "prescribed" and re.search(r"(?i)\b(?:first|initially|originally|initial)\b", excerpt) else
                "unspecified"
            ),
        ))
    return results, ambiguous


def _requested_event_role(question: str, event_type: str) -> str | None:
    if event_type == "renewed":
        return "renewal"
    if event_type == "discontinued":
        return "discontinuation"
    if event_type == "restarted":
        return "restart"
    if event_type == "prescribed" and re.search(r"(?i)\b(?:first|initial|initially|original|originally)\b", question):
        return "original"
    return None


def _event_label(event_type: str, wording: str = "explicit") -> str:
    if event_type == "prescribed" and wording == "requested_alias":
        return "prescription/order"
    return {
        "prescribed": "prescribed",
        "renewed": "renewed/refilled",
        "dispensed": "dispensed",
        "picked_up": "picked up",
        "started": "started",
        "discontinued": "discontinued",
        "restarted": "restarted",
    }[event_type]


def _medication_options(names: tuple[str, ...], resolution: Any) -> list[dict[str, Any]]:
    referent = getattr(resolution, "referent", {}) or {}
    ordered: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in referent.get("items", []) if isinstance(referent, dict) else []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("value") or "").casefold()
        match = next(
            (name for name in names if name == label or re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", label, re.I)),
            None,
        )
        if match and match not in used:
            ordered.append(dict(item))
            used.add(match)
    for name in names:
        if name not in used:
            ordered.append({"label": name, "value": name})
            used.add(name)
    return ordered


def _event_clarification_options() -> list[dict[str, Any]]:
    return [
        {"label": "dispensed", "value": "dispensed", "event_type": "dispensed"},
        {"label": "picked up", "value": "picked up", "event_type": "picked_up"},
    ]


def search_medication_event(
    db: sqlite3.Connection,
    *,
    patient_id: str,
    question: str,
    resolution: Any,
    history: list[dict[str, Any]] | None = None,
) -> MedicationEventResult | None:
    """Return a medication-event result, or None when this route does not apply."""
    sources = load_eligible_sources(db, patient_id)
    # Provenance facts are a derived encrypted index. Some focused unit tests
    # exercise this module against a minimal SQLite fixture without the full
    # application schema, so retain the conservative parser as a compatibility
    # fallback when the fact-store tables do not exist.
    has_fact_store = bool(db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='clinical_fact_sources'"
    ).fetchone())
    if has_fact_store:
        from .fact_store import (
            medication_event_facts, medication_event_resolution, medication_names,
            sync_patient_facts, validate_fact_provenance,
        )
        sync_patient_facts(db, patient_id)
        known = medication_names(db, patient_id) or _known_medications(sources)
    else:
        medication_event_facts = None
        medication_event_resolution = None
        validate_fact_provenance = None
        sync_patient_facts = None
        known = _known_medications(sources)
    if not _is_medication_event_query(question, resolution, known):
        return None
    if not sources:
        return MedicationEventResult(
            handled=True,
            status="no_reviewed_sources",
            answer="No approved, context-enabled records are available for this medication-event search.",
            coverage="no_eligible_sources",
            eligible_source_count=0,
            searched_source_count=0,
            source_snapshot={},
        )

    total_chars = sum(len(source.body) for source in sources)
    snapshot = source_snapshot(sources)
    if len(sources) > MAX_SOURCES or total_chars > MAX_SOURCE_CHARACTERS:
        return MedicationEventResult(
            handled=True,
            status="source_budget_exceeded",
            answer=(
                "The reviewed source set is too large for a complete medication-event search in this request. "
                "The search was not completed, so I can’t claim that the date is absent."
            ),
            coverage="not_examined_due_to_budget",
            eligible_source_count=len(sources),
            searched_source_count=0,
            source_snapshot=snapshot,
        )
    if len(known) > MAX_MEDICATION_CANDIDATES:
        return MedicationEventResult(
            handled=True,
            status="entity_budget_exceeded",
            answer=(
                "The chart contains more medication candidates than this medication-event search can examine safely. "
                "Please name the medication; no absence claim is being made."
            ),
            coverage="not_examined_due_to_budget",
            eligible_source_count=len(sources),
            searched_source_count=0,
            source_snapshot=snapshot,
        )

    medications = _selected_medications(question, known, resolution, history)
    event_type, wording = _explicit_event_from_resolution(question, resolution)
    ranked_ids = _rank_sources_fts(sources, question)

    # "Received" is not a single medication event. Preserve the selected
    # medication(s) while asking whether the user means dispensing or pickup.
    if _RECEIVED_WORDING.search(question) and event_type is None:
        options = _event_clarification_options()
        if not medications:
            meds = _medication_options(known, resolution)
            if len(meds) > 1:
                return MedicationEventResult(
                    handled=True,
                    status="needs_medication",
                    answer="Which medication do you mean—" + " or ".join(str(item.get("label")) for item in meds[:4]) + "?",
                    coverage="not_applicable",
                    clarification_kind="medication",
                    clarification_question="Which medication do you mean?",
                    clarification_options=meds[:4],
                    eligible_source_count=len(sources),
                    searched_source_count=0,
                    source_snapshot=snapshot,
                    ranked_source_ids=ranked_ids,
                )
        return MedicationEventResult(
            handled=True,
            status="needs_event",
            answer="For the medication date, do you mean when it was dispensed or when the patient picked it up?",
            coverage="not_applicable",
            medications=medications,
            clarification_kind="event",
            clarification_question="Do you mean dispensed or picked up?",
            clarification_options=options,
            eligible_source_count=len(sources),
            searched_source_count=0,
            source_snapshot=snapshot,
            ranked_source_ids=ranked_ids,
        )

    if not event_type:
        if _DATE_QUESTION.search(question) or _REQUESTED_WORDING.search(question):
            return MedicationEventResult(
                handled=True,
                status="needs_event",
                answer="Do you mean the prescription/order date, the dispensing date, the pickup date, or when the medication was started?",
                coverage="not_applicable",
                medications=medications,
                clarification_kind="event",
                clarification_question="Which medication event date do you mean?",
                clarification_options=[
                    {"label": "prescribed/ordered", "value": "prescribed", "event_type": "prescribed"},
                    {"label": "renewed/refilled", "value": "renewed", "event_type": "renewed"},
                    {"label": "dispensed", "value": "dispensed", "event_type": "dispensed"},
                    {"label": "picked up", "value": "picked up", "event_type": "picked_up"},
                    {"label": "started", "value": "started", "event_type": "started"},
                    {"label": "discontinued", "value": "discontinued", "event_type": "discontinued"},
                    {"label": "restarted", "value": "restarted", "event_type": "restarted"},
                ],
                eligible_source_count=len(sources),
                searched_source_count=0,
                source_snapshot=snapshot,
                ranked_source_ids=ranked_ids,
            )
        return None

    if not medications:
        options = _medication_options(known, resolution)
        return MedicationEventResult(
            handled=True,
            status="needs_medication",
            answer=(
                "Which medication do you mean? "
                + (("Reviewed medication names: " + ", ".join(str(item.get("label")) for item in options[:8]) + ".") if options else "I found no reliably named medication in the reviewed sources.")
            ),
            coverage="not_applicable",
            event_type=event_type,
            clarification_kind="medication",
            clarification_question="Which medication do you mean?",
            clarification_options=options[:8],
            eligible_source_count=len(sources),
            searched_source_count=0,
            source_snapshot=snapshot,
            ranked_source_ids=ranked_ids,
        )

    task = MedicationEventTask(
        medications=medications,
        event_type=event_type,
        wording=wording,
    )
    requested_role = _requested_event_role(question, task.event_type)
    indexed_facts = medication_event_facts(db, patient_id, task.medications, task.event_type) if has_fact_store else []
    if indexed_facts and validate_fact_provenance is not None and not validate_fact_provenance(db, indexed_facts):
        # Never trust a derived row after source/version/hash drift. A resync is
        # attempted once; if provenance still fails, the request is incomplete.
        assert sync_patient_facts is not None and medication_event_facts is not None and validate_fact_provenance is not None
        sync_patient_facts(db, patient_id)
        indexed_facts = medication_event_facts(db, patient_id, task.medications, task.event_type)
        if indexed_facts and not validate_fact_provenance(db, indexed_facts):
            return MedicationEventResult(
                handled=True,
                status="provenance_validation_failed",
                answer="The medication-event index no longer matches the current reviewed source. Refresh the chart and retry; no date claim was returned.",
                coverage="provenance_recheck_failed",
                medications=task.medications,
                event_type=task.event_type,
                eligible_source_count=len(sources),
                searched_source_count=0,
                source_snapshot=snapshot,
                ranked_source_ids=ranked_ids,
            )

    active_indexed_facts: list[Any] = []
    conflict_details: list[tuple[str, list[Any], list[str]]] = []
    if has_fact_store and medication_event_resolution is not None:
        by_id = {fact.id: fact for fact in indexed_facts}
        for medication in task.medications:
            resolutions = medication_event_resolution(
                db, patient_id, medication, task.event_type, event_role=requested_role
            )
            for resolution_row in resolutions:
                active_ids = [
                    fact_id for fact_id in resolution_row.fact_ids
                    if fact_id not in set(resolution_row.superseded_fact_ids)
                ]
                active = [by_id[fact_id] for fact_id in active_ids if fact_id in by_id]
                if resolution_row.status == "conflict":
                    conflict_details.append((medication, active, list(resolution_row.values.get("event_dates") or [])))
                elif resolution_row.preferred_fact_id and resolution_row.preferred_fact_id in by_id:
                    active_indexed_facts.append(by_id[resolution_row.preferred_fact_id])
                elif active:
                    # Consistent duplicate assertions share one date; one exact
                    # source quote is enough for the answer while the resolution
                    # object retains all supporting fact IDs.
                    active_indexed_facts.append(active[0])
    else:
        active_indexed_facts = indexed_facts

    if conflict_details:
        citations = [
            MedicationEventCitation(
                source_id=fact.source_id, source_version=fact.source_version, source_title=fact.source_title,
                start=fact.start_offset, end=fact.end_offset, quote=fact.quote,
                medication=(fact.entity_label or "").casefold(), event_type=fact.event_type or task.event_type,
                event_date=fact.event_date or "", event_role=fact.event_role,
            )
            for _, facts, _ in conflict_details for fact in facts if fact.event_date
        ]
        pieces = [
            f"{medication}: conflicting { _event_label(task.event_type, task.wording) } dates are documented ({', '.join(dates)})"
            for medication, _, dates in conflict_details
        ]
        return MedicationEventResult(
            handled=True, status="conflicting_documentation",
            answer="; ".join(pieces) + ". No date was selected as authoritative; review the cited source versions.",
            coverage="provenance_fact_index_conflict_checked", citations=citations[:MAX_CITATIONS],
            medications=task.medications, event_type=task.event_type, eligible_source_count=len(sources),
            searched_source_count=len(sources), source_snapshot=snapshot, ranked_source_ids=ranked_ids,
        )

    citations: list[MedicationEventCitation] = [
        MedicationEventCitation(
            source_id=fact.source_id,
            source_version=fact.source_version,
            source_title=fact.source_title,
            start=fact.start_offset,
            end=fact.end_offset,
            quote=fact.quote,
            medication=(fact.entity_label or "").casefold(),
            event_type=fact.event_type or task.event_type,
            event_date=fact.event_date or "",
            event_role=fact.event_role,
        )
        for fact in active_indexed_facts
        if fact.event_date
    ]
    # The persistent index stores only explicit supported events. We still run
    # the conservative ambiguity detector when no explicit event fact exists so
    # uncertain/negated wording cannot be mislabeled as an absent date.
    ambiguous = False
    if not citations:
        for medication in task.medications:
            for source in sources:
                hits, uncertain = _event_mentions(source, medication, task.event_type, known)
                if not has_fact_store:
                    citations.extend(hits)
                ambiguous = ambiguous or uncertain

    event_label = _event_label(task.event_type, task.wording)
    if ambiguous:
        return MedicationEventResult(
            handled=True,
            status="needs_review",
            answer=(
                "Some reviewed passages mention " + ", ".join(task.medications)
                + f" and the {event_label} event, but the wording is ambiguous, negated, uncertain, or contains multiple event dates. "
                "Review the cited source before assigning a date."
            ),
            coverage="provenance_fact_index_plus_ambiguity_scan",
            citations=citations[:MAX_CITATIONS],
            medications=task.medications,
            event_type=task.event_type,
            eligible_source_count=len(sources),
            searched_source_count=len(sources),
            source_snapshot=snapshot,
            ranked_source_ids=ranked_ids,
        )

    unique = sorted({(citation.medication, citation.event_date) for citation in citations})
    statements: list[str] = []
    for medication in task.medications:
        dates = sorted({event_date for name, event_date in unique if name == medication})
        if not dates:
            statements.append(f"{medication}: no explicit {event_label} date found in the reviewed sources")
        elif len(dates) == 1:
            statements.append(f"{medication}: a reviewed source states {event_label} on {dates[0]}")
        else:
            if task.event_type in {"renewed", "dispensed", "picked_up", "started", "restarted", "discontinued"}:
                statements.append(
                    f"{medication}: multiple {event_label} dates are documented ({', '.join(dates)}); they are retained as separate documented events"
                )
            else:
                statements.append(
                    f"{medication}: multiple {event_label} dates are documented ({', '.join(dates)}); these may represent separate orders/events or conflicting documentation"
                )

    if not unique:
        answer = (
            "I searched all " + str(len(sources)) + " eligible reviewed source"
            + ("" if len(sources) == 1 else "s")
            + ". No explicit " + event_label + " date was found for " + ", ".join(task.medications)
            + " in the supported medication-event date patterns. "
            "Visit dates and report dates were not substituted for the medication event."
        )
        status = "no_explicit_date_found"
    else:
        answer = "; ".join(statements) + ". These are reviewed source statements, not independently verified pharmacy events."
        counts = [
            len({event_date for name, event_date in unique if name == medication})
            for medication in task.medications
        ]
        if all(count == 1 for count in counts):
            status = "documented_in_source"
        elif task.event_type in {"renewed", "dispensed", "picked_up", "started", "restarted", "discontinued"} and all(count >= 1 for count in counts):
            status = "documented_event_series"
        else:
            status = "multiple_or_partial"

    return MedicationEventResult(
        handled=True,
        status=status,
        answer=answer,
        coverage="provenance_fact_index_current_sources",
        citations=citations[:MAX_CITATIONS],
        medications=task.medications,
        event_type=task.event_type,
        eligible_source_count=len(sources),
        searched_source_count=len(sources),
        source_snapshot=snapshot,
        ranked_source_ids=ranked_ids,
    )


def event_sources_for_ui(
    result: MedicationEventResult,
    all_sources: list[EligibleSource],
) -> list[dict[str, Any]]:
    """Build compact existing-UI source objects from event evidence/search hits."""
    lookup = {source.id: source for source in all_sources}
    citation_by_source: dict[str, MedicationEventCitation] = {}
    for citation in result.citations or []:
        citation_by_source.setdefault(citation.source_id, citation)
    ordered_ids: list[str] = []
    for source_id in result.source_ids + list(result.ranked_source_ids or []):
        if source_id in lookup and source_id not in ordered_ids:
            ordered_ids.append(source_id)
    if not ordered_ids and result.medications:
        for source in all_sources:
            if any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", source.body, re.I) for name in result.medications):
                ordered_ids.append(source.id)
                if len(ordered_ids) >= 3:
                    break
    output: list[dict[str, Any]] = []
    counts = {"R": 0, "N": 0}
    for source_id in ordered_ids[:4]:
        source = lookup[source_id]
        counts[source.kind] += 1
        citation = citation_by_source.get(source_id)
        if citation:
            excerpt = citation.quote
        else:
            # Prefer a medication-containing sentence/snippet for missing-date answers.
            excerpt = source.body[:900]
            for medication in result.medications:
                match = re.search(r"[^\n.!?;]*\b" + re.escape(medication) + r"\b[^\n.!?;]*(?:[.!?;]|$)", source.body, re.I)
                if match:
                    excerpt = match.group(0).strip()
                    break
        output.append({
            "label": f"{source.kind}{counts[source.kind]}",
            "id": source.id,
            "title": source.title,
            "version": source.version,
            "excerpt": excerpt[:900],
            "partial": len(source.body) > len(excerpt),
            "score": 100 if citation else 10,
            "kind": source.kind,
            "record_date": source.record_date or source.updated_at,
            "document_type": None,
        })
    return output


def validate_source_snapshot(
    expected: dict[str, tuple[int, str]],
    current_sources: list[EligibleSource],
) -> bool:
    return expected == source_snapshot(current_sources)


def apply_workflow_clarification(resolution: Any, result: MedicationEventResult) -> None:
    """Mutate the existing dialogue resolution so clarification persists safely."""
    if not result.clarification_kind:
        return
    resolution.intent = INTENT_MEDICATIONS
    options = [dict(item) for item in (result.clarification_options or [])]
    prior_referent = dict(getattr(resolution, "referent", {}) or {})
    medication_items = [item for item in prior_referent.get("items", []) if isinstance(item, dict)]
    if result.medications and not medication_items:
        medication_items = [{"label": name, "value": name} for name in result.medications]
    if result.clarification_kind == "event":
        resolution.referent = {
            "type": "medication_event_choice",
            "items": options,
            "medication_items": medication_items,
            "selected_medications": list(result.medications),
            "source_ids": prior_referent.get("source_ids", []),
        }
        resolution.relation = "medication_event_date"
    else:
        resolution.referent = {
            "type": "medication",
            "items": options,
            "source_ids": prior_referent.get("source_ids", []),
            "event_type": result.event_type,
        }
        # Preserve the event date request while asking which medication.
        resolution.relation = {
            "prescribed": "prescribed_date",
            "dispensed": "dispensed_date",
            "picked_up": "pickup_date",
            "started": "started_date",
        }.get(result.event_type or "", getattr(resolution, "relation", "new_question"))
    resolution.clarification_required = True
    resolution.clarification_question = result.clarification_question or result.answer
    resolution.reason = f"medication_event_{result.clarification_kind}_clarification"
    resolution.confidence = min(float(getattr(resolution, "confidence", 0.8)), 0.75)


def event_type_from_resolution(resolution: Any) -> str | None:
    referent = getattr(resolution, "referent", {}) or {}
    selected = referent.get("selected") if isinstance(referent, dict) else None
    if isinstance(selected, dict) and selected.get("event_type"):
        return str(selected["event_type"])
    return str(referent.get("event_type")) if isinstance(referent, dict) and referent.get("event_type") else None

_EVENT_RELATION_BY_TYPE = {
    "prescribed": "prescribed_date",
    "dispensed": "dispensed_date",
    "picked_up": "pickup_date",
    "started": "started_date",
}


def stamp_event_resolution(resolution: Any, result: MedicationEventResult) -> None:
    """Persist the resolved medication-event task in the existing dialogue state."""
    resolution.intent = INTENT_MEDICATIONS
    if result.event_type:
        resolution.relation = _EVENT_RELATION_BY_TYPE.get(result.event_type, resolution.relation)
        resolution.reason = f"medication_event_{result.status}"
    referent = dict(getattr(resolution, "referent", {}) or {})
    if result.medications:
        items = [item for item in referent.get("items", []) if isinstance(item, dict)]
        selected_items = []
        for name in result.medications:
            match = next((item for item in items if str(item.get("label") or item.get("value") or "").casefold() == name), None)
            selected_items.append(match or {"label": name, "value": name})
        if len(selected_items) == 1:
            referent["selected"] = selected_items[0]
        else:
            referent["selected"] = {
                "label": "both" if len(selected_items) == 2 else "all",
                "value": "both" if len(selected_items) == 2 else "all",
                "selection": "all",
                "items": selected_items,
            }
        referent["selected_medications"] = list(result.medications)
    if result.event_type:
        referent["event_type"] = result.event_type
    resolution.referent = referent
    resolution.confidence = max(float(getattr(resolution, "confidence", 0.0)), 0.97)
