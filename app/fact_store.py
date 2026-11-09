"""Encrypted, provenance-backed derived fact index for reviewed chart sources.

The reviewed document/note remains authoritative. Facts are a deterministic
search index that always retains exact source/version/hash/quote provenance.
Stale rows are deleted when the source changes, is reassigned, or leaves model
context. No model is used for extraction in this increment.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import re
import sqlite3
from typing import Any, Iterable

from .fact_normalization import (
    normalize_allergy, normalize_condition, normalize_measurement, normalize_medication,
    normalize_pending, normalize_plan,
)
from .medication_event_workflow import (
    EligibleSource,
    _event_mentions,
    _known_medications,
    load_eligible_sources,
)
from .vault import now, uid

EXTRACTION_VERSION = "provenance-facts-v3"
EXTRACTION_METHOD = "deterministic_rules"
SEMANTIC_VERSION = "fact-semantics-v1"
FACT_TYPES = {
    "medication",
    "medication_event",
    "condition",
    "allergy",
    "pending_item",
    "measurement",
    "plan",
}
EVENT_TYPES = ("prescribed", "renewed", "dispensed", "picked_up", "started", "discontinued", "restarted")


@dataclass(frozen=True)
class Fact:
    id: str
    patient_id: str
    session_id: str | None
    fact_type: str
    entity_key: str | None
    entity_label: str | None
    event_type: str | None
    event_role: str
    value_text: str
    normalized: dict[str, Any]
    event_date: str | None
    asserted_at: str | None
    effective_start: str | None
    effective_end: str | None
    assertion_key: str
    supersedes: list[dict[str, Any]]
    status: str
    polarity: str
    certainty: str
    source_kind: str
    source_id: str
    source_version: int
    source_sha256: str
    source_title: str
    quote: str
    start_offset: int
    end_offset: int
    extraction_method: str = EXTRACTION_METHOD
    extraction_version: str = EXTRACTION_VERSION
    semantic_version: str = SEMANTIC_VERSION

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fact_type": self.fact_type,
            "entity_key": self.entity_key,
            "entity_label": self.entity_label,
            "event_type": self.event_type,
            "event_role": self.event_role,
            "value_text": self.value_text,
            "normalized": self.normalized,
            "event_date": self.event_date,
            "asserted_at": self.asserted_at,
            "effective_start": self.effective_start,
            "effective_end": self.effective_end,
            "assertion_key": self.assertion_key,
            "supersedes": self.supersedes,
            "status": self.status,
            "polarity": self.polarity,
            "certainty": self.certainty,
            "source": {
                "kind": self.source_kind,
                "id": self.source_id,
                "version": self.source_version,
                "sha256": self.source_sha256,
                "title": self.source_title,
                "quote": self.quote,
                "start": self.start_offset,
                "end": self.end_offset,
            },
            "extraction_method": self.extraction_method,
            "extraction_version": self.extraction_version,
            "semantic_version": self.semantic_version,
        }


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _key(value: str) -> str:
    return re.sub(r"\W+", "-", value.casefold()).strip("-")[:120]


def _lines(text: str) -> Iterable[tuple[int, int, str]]:
    for match in re.finditer(r"[^\n]+", text):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        start = match.start() + leading
        end = match.start() + trailing
        if end > start:
            yield start, end, text[start:end]


def _sentences(text: str) -> Iterable[tuple[int, int, str]]:
    for line_start, _, line in _lines(text):
        for match in re.finditer(r"[^.!?]+(?:[.!?]|$)", line):
            raw = match.group(0)
            candidate = raw.strip()
            if len(candidate) < 8:
                continue
            leading = len(raw) - len(raw.lstrip())
            start = line_start + match.start() + leading
            end = start + len(candidate)
            yield start, end, text[start:end]


def _split_list(value: str) -> list[str]:
    text = " ".join(value.split()).strip(" .;:")
    if not text:
        return []
    parts = re.split(r"\s*;\s*|\s+and\s+(?=[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z0-9-]+){0,3}\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b)", text, flags=re.I)
    return [part.strip(" .;:") for part in parts if part.strip(" .;:")]


def _medication_label(value: str) -> str:
    clean = " ".join(value.split()).strip(" .;:")
    match = re.match(r"^(.+?)(?=\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b|,\s*(?:one|two|three|\d)|$)", clean, re.I)
    return (match.group(1) if match else clean).strip(" ,.;:")



def _pending_label(value: str) -> str:
    clean = " ".join(str(value or "").split()).strip(" .;:")
    clean = re.sub(r"(?i)^(?:the\s+)?(?:(?:requested|pending)\s+)+", "", clean).strip()
    clean = re.sub(r"(?i)\s+(?:remains?|is|still)\s+pending$", "", clean).strip()
    return clean or "pending item"


def _pending_identity(value: str) -> str:
    clean = _pending_label(value)
    clean = re.sub(r"(?i)^(?:(?:outside|external)\s+)+", "", clean).strip()
    return clean or "pending item"


def _allergy_key(value: str) -> str:
    clean = " ".join(str(value or "").split()).strip(" .;:")
    match = re.match(r"(?i)^(.+?)\s+(?:allergy\b|caused\b)", clean)
    return (match.group(1) if match else clean).strip(" .;:")


def _entity_identity(fact_type: str, label: str, value: str) -> str:
    if fact_type == "pending_item":
        return _pending_identity(label or value)
    if fact_type == "allergy":
        return _allergy_key(label or value)
    return label or value


_DATE_PATTERN = r"(?:\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})"


def _normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    raw = str(value).strip()
    try:
        return date.fromisoformat(raw[:10]).isoformat()
    except ValueError:
        try:
            return datetime.strptime(raw.replace(",", ""), "%B %d %Y").date().isoformat()
        except ValueError:
            return None


def _source_record_date(body: str) -> str | None:
    patterns = [
        r"(?im)^\s*(?:report|document|visit|encounter|service)\s+date\s*:\s*(\d{4}-\d{2}-\d{2})\s*$",
        r"(?im)\b(?:most\s+recent\s+reviewed\s+visit|visit|review)\s*\(?\s*(\d{4}-\d{2}-\d{2})\s*\)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, body)
        if match:
            return _normalize_date(match.group(1))
    return None


def _event_role(event_type: str | None, quote: str, normalized: dict[str, Any]) -> str:
    if event_type == "renewed":
        return "renewal"
    if event_type == "discontinued":
        return "discontinuation"
    if event_type == "restarted":
        return "restart"
    if event_type == "prescribed":
        if re.search(r"(?i)\b(?:first|initially|originally|initial)\b", quote):
            return "original"
        if normalized.get("order_id"):
            return "order"
    return "unspecified"


def _assertion_key(
    fact_type: str, entity_key: str | None, event_type: str | None, event_role: str,
    normalized: dict[str, Any], effective_start: str | None,
) -> str:
    order_id = str(normalized.get("order_id") or "").casefold().strip()
    if fact_type == "medication_event":
        return "|".join(["medication_event", entity_key or "", event_type or "", event_role or "unspecified", order_id])
    if fact_type == "measurement":
        name = entity_key or _key(str(normalized.get("name") or "measurement"))
        return "|".join(["measurement", name, effective_start or "undated"])
    return "|".join([fact_type, entity_key or "", event_type or "", event_role or "unspecified"])


def _order_id(text: str) -> str | None:
    match = re.search(r"(?i)\b(?:order|prescription|rx)\s*(?:id|number|#)\s*[:#-]?\s*([A-Za-z0-9._-]{2,40})", text)
    return match.group(1) if match else None


def _correction_medication_event(line: str) -> dict[str, Any] | None:
    verbs = {
        "prescribed": r"prescribed|ordered",
        "renewed": r"renewed|refilled|reordered",
        "dispensed": r"dispensed|filled",
        "picked_up": r"picked[ -]?up|collected",
        "started": r"started|began",
        "discontinued": r"discontinued|stopped",
        "restarted": r"restarted|resumed",
    }
    for event_type, verb in verbs.items():
        patterns = [
            rf"(?i)\b(?:correction|corrected)\s*:\s*([A-Za-z][A-Za-z-]{{2,35}})\s+(?:was\s+)?(?:{verb})\s+(?:on\s+)?({_DATE_PATTERN}).*?\b(?:prior|previous|old)\s+(?:date\s+)?({_DATE_PATTERN})\s+(?:was\s+)?(?:incorrect|wrong|entered\s+in\s+error)",
            rf"(?i)\b(?:correction|corrected)\s*:\s*(?:the\s+)?(?:{verb}|prescription|dispensing|pickup|start|discontinuation|restart)\s+date\s+(?:for|of)\s+([A-Za-z][A-Za-z-]{{2,35}})\s+(?:is|was)\s+({_DATE_PATTERN}).*?\b(?:prior|previous|old)\s+(?:date\s+)?({_DATE_PATTERN})\s+(?:was\s+)?(?:incorrect|wrong|entered\s+in\s+error)",
        ]
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                new_date = _normalize_date(match.group(2))
                old_date = _normalize_date(match.group(3))
                if new_date and old_date:
                    return {
                        "medication": match.group(1).casefold(),
                        "event_type": event_type,
                        "event_date": new_date,
                        "supersedes_date": old_date,
                    }
    return None


def _correction_measurement(line: str) -> dict[str, Any] | None:
    patterns = [
        ("creatinine", r"\d+(?:\.\d+)?\s*mg/dL"),
        ("hemoglobin A1c", r"\d+(?:\.\d+)?\s*percent"),
        ("LDL cholesterol", r"\d+(?:\.\d+)?\s*mg/dL"),
        ("blood pressure", r"\d{2,3}/\d{2,3}\s*mmHg"),
    ]
    for label, value_pattern in patterns:
        match = re.search(
            rf"(?i)\b(?:correction|corrected)\s*:\s*{re.escape(label)}\s+(?:was|is)\s+({value_pattern})\s+(?:on\s+)?({_DATE_PATTERN}).*?\b(?:prior|previous|old)\s+value\s+({value_pattern})\s+(?:was\s+)?(?:incorrect|wrong|entered\s+in\s+error)",
            line,
        )
        if match:
            event_date = _normalize_date(match.group(2))
            if event_date:
                return {
                    "label": label,
                    "value": " ".join(match.group(1).split()),
                    "event_date": event_date,
                    "supersedes_value": " ".join(match.group(3).split()),
                }
    return None

def _add(
    facts: list[Fact], source: EligibleSource, patient_id: str, session_id: str | None,
    *, fact_type: str, value: str, quote: str, start: int, end: int,
    entity_label: str | None = None, event_type: str | None = None,
    event_date: str | None = None, normalized: dict[str, Any] | None = None,
    event_role: str | None = None, asserted_at: str | None = None,
    effective_start: str | None = None, effective_end: str | None = None,
    supersedes: list[dict[str, Any]] | None = None,
    status: str = "current", polarity: str = "affirmed", certainty: str = "explicit",
) -> None:
    value = " ".join(str(value).split()).strip(" .;")
    if not value or fact_type not in FACT_TYPES:
        return
    label = " ".join(str(entity_label or value).split()).strip(" .;")
    if fact_type == "pending_item":
        label = _pending_label(label)
    identity = _entity_identity(fact_type, label, value)
    entity_key = _key(identity) if identity else None
    normalized_value = dict(normalized or {})
    if fact_type == "medication_event":
        order_id = _order_id(quote)
        if order_id and not normalized_value.get("order_id"):
            normalized_value["order_id"] = order_id.casefold()
    normalized_event_date = _normalize_date(event_date) or event_date
    role = event_role or _event_role(event_type, quote, normalized_value)
    asserted = asserted_at if asserted_at is not None else source.record_date
    assertion_key = _assertion_key(
        fact_type, entity_key, event_type, role, normalized_value, effective_start or normalized_event_date
    )
    source_hash = _sha(source.body)
    facts.append(Fact(
        id=uid(), patient_id=patient_id, session_id=session_id,
        fact_type=fact_type, entity_key=entity_key, entity_label=label or None,
        event_type=event_type, event_role=role, value_text=value,
        normalized=normalized_value, event_date=normalized_event_date, asserted_at=asserted,
        effective_start=_normalize_date(effective_start) or effective_start,
        effective_end=_normalize_date(effective_end) or effective_end,
        assertion_key=assertion_key, supersedes=list(supersedes or []),
        status=status, polarity=polarity, certainty=certainty,
        source_kind="document" if source.kind == "R" else "note",
        source_id=source.id, source_version=source.version, source_sha256=source_hash,
        source_title=source.title, quote=quote, start_offset=start, end_offset=end,
    ))


def extract_facts(source: EligibleSource, patient_id: str, session_id: str | None = None) -> list[Fact]:
    facts: list[Fact] = []
    body = source.body

    # Generic structured observation/result blocks. This broadens coverage beyond
    # a short hard-coded list while retaining exact source offsets and optional
    # LOINC/synthetic codes. It intentionally requires the paired labels rather
    # than trying to infer arbitrary numbers from free prose.
    line_rows = list(_lines(body))
    for index, (start, _, quote) in enumerate(line_rows):
        observation = re.match(r"(?i)^\s*Observation\s*:\s*(.+?)\s*$", quote)
        if not observation:
            continue
        if index + 1 >= len(line_rows):
            continue
        result_row = line_rows[index + 1]
        result = re.match(r"(?i)^\s*Result\s*:\s*(.+?)\s*$", result_row[2])
        if not result:
            continue
        label = " ".join(observation.group(1).split()).strip(" .;:")
        value = " ".join(result.group(1).split()).strip(" .;:")
        code = None
        result_status = None
        end = result_row[1]
        for lookahead in line_rows[index + 2:index + 6]:
            code_match = re.match(r"(?i)^\s*(?:LOINC|LOINC or synthetic test code)\s*:\s*(.+?)\s*$", lookahead[2])
            status_match = re.match(r"(?i)^\s*Result status\s*:\s*(.+?)\s*$", lookahead[2])
            if code_match:
                code = code_match.group(1).strip()
                end = lookahead[1]
            if status_match:
                result_status = status_match.group(1).strip(" .")
                end = lookahead[1]
        combined_quote = body[start:end]
        _add(
            facts, source, patient_id, session_id, fact_type="measurement",
            value=value, entity_label=label, quote=combined_quote, start=start, end=end,
            effective_start=source.record_date,
            normalized=normalize_measurement(label, value, code=code, status=result_status),
        )

    # Whole-line structured headings.
    for start, end, quote in _lines(body):
        line = " ".join(quote.split())
        line = re.sub(r"^[-*]\s*", "", line)
        m = re.match(r"(?i)^Current medications?\s*:\s*(.+)$", line)
        if m:
            for item in _split_list(m.group(1)):
                _add(facts, source, patient_id, session_id, fact_type="medication", value=item,
                     entity_label=_medication_label(item), quote=quote, start=start, end=end,
                     normalized=normalize_medication(item, status="current"))
        m = re.match(r"(?i)^(?:Historical|Past|Previous) medications?\s*:\s*(.+)$", line)
        if m:
            for item in _split_list(m.group(1)):
                _add(facts, source, patient_id, session_id, fact_type="medication", value=item,
                     entity_label=_medication_label(item), quote=quote, start=start, end=end,
                     normalized=normalize_medication(item, status="historical"), status="historical")
        m = re.match(r"(?i)^(?:Discontinued|Stopped) medications?\s*:\s*(.+)$", line)
        if m:
            for item in _split_list(m.group(1)):
                _add(facts, source, patient_id, session_id, fact_type="medication", value=item,
                     entity_label=_medication_label(item), quote=quote, start=start, end=end,
                     normalized=normalize_medication(item, status="discontinued"), status="discontinued")
        m = re.match(r"(?i)^Active problems?\s*:\s*(.+)$", line)
        if m:
            values = [v.strip(" .;") for v in re.split(r"\s*;\s*|\s+and\s+", m.group(1)) if v.strip(" .;")]
            for item in values:
                code_match = re.search(r"\(([A-Z][A-Z0-9.]{1,15})\)\s*$", item)
                display = re.sub(r"\s*\([A-Z][A-Z0-9.]{1,15}\)\s*$", "", item).strip()
                _add(facts, source, patient_id, session_id, fact_type="condition", value=display,
                     entity_label=display, quote=quote, start=start, end=end,
                     normalized=normalize_condition(display, code=code_match.group(1) if code_match else None))
        m = re.match(
            r"(?i)^(?:Active\s+documented\s+condition|Documented\s+(?:condition|problem)|Active\s+problem\s+documented)\s*:\s*(.+)$",
            line,
        )
        if m:
            raw_value = m.group(1).strip(" .;")
            code_match = re.search(r"\(([A-Z][A-Z0-9.]{1,15})\)\s*$", raw_value)
            value = re.sub(r"\s*\([A-Z][A-Z0-9.]{1,15}\)\s*$", "", raw_value).strip(" .;")
            if value:
                _add(facts, source, patient_id, session_id, fact_type="condition", value=value,
                     entity_label=value, quote=quote, start=start, end=end,
                     normalized=normalize_condition(value, code=code_match.group(1) if code_match else None))
        m = re.match(r"(?i)^(?:Medication\s+appearing\s+in\s+(?:source\s+)?records|Medication\s+documented|Documented\s+medication)\s*:\s*(.+)$", line)
        if m:
            value = m.group(1).strip(" .;")
            if value and not re.search(r"(?i)^No scheduled medication documented", value):
                _add(facts, source, patient_id, session_id, fact_type="medication", value=value,
                     entity_label=_medication_label(value), quote=quote, start=start, end=end,
                     normalized=normalize_medication(value, status="documented_in_records"), status="documented")
        m = re.match(r"(?i)^Allergy statement\s*:\s*(.+)$", line)
        if m:
            value = m.group(1).strip()
            polarity = "none_documented" if re.search(r"(?i)\bno known drug allergies\b|\bnone\b", value) else "affirmed"
            _add(facts, source, patient_id, session_id, fact_type="allergy", value=value,
                 entity_label=value, quote=quote, start=start, end=end, polarity=polarity,
                 normalized=normalize_allergy(value))
        m = re.match(r"(?i)^Medication allerg(?:y|ies)\s*:\s*(.+)$", line)
        if m:
            value = m.group(1).strip()
            polarity = "none_documented" if re.search(r"(?i)\bnone\b|no\s+.*documented", value) else "affirmed"
            _add(facts, source, patient_id, session_id, fact_type="allergy", value=value,
                 entity_label=value, quote=quote, start=start, end=end, polarity=polarity,
                 normalized=normalize_allergy(value))
        m = re.match(r"(?i)^Medication allergy\s*:\s*(.+)$", line)
        if m:
            value = m.group(1).strip()
            _add(facts, source, patient_id, session_id, fact_type="allergy", value=value,
                 entity_label=value, quote=quote, start=start, end=end,
                 normalized=normalize_allergy(value))
        m = re.match(r"(?i)^Documented plan\s*:\s*(.+)$", line)
        if m:
            _add(facts, source, patient_id, session_id, fact_type="plan", value=m.group(1),
                 entity_label="documented plan", quote=quote, start=start, end=end,
                 normalized=normalize_plan(m.group(1)))
        correction_measurement = _correction_measurement(line)
        if correction_measurement:
            _add(
                facts, source, patient_id, session_id, fact_type="measurement",
                value=correction_measurement["value"], entity_label=correction_measurement["label"],
                quote=quote, start=start, end=end, effective_start=correction_measurement["event_date"],
                normalized={
                    "name": correction_measurement["label"],
                    "value": correction_measurement["value"],
                    "is_correction": True,
                },
                supersedes=[{
                    "value": correction_measurement["supersedes_value"],
                    "effective_start": correction_measurement["event_date"],
                }],
            )
        measurement_patterns = [
            ("blood pressure", r"(?i)\bblood pressure\s+(?:was\s+)?(\d{2,3}/\d{2,3}\s*mmHg)"),
            ("pulse", r"(?i)\bpulse\s+(?:was\s+)?(\d{2,3}\s*(?:beats/min|bpm)?)"),
            ("temperature", r"(?i)\btemperature\s+(?:was\s+)?(\d{2,3}(?:\.\d+)?\s*F)"),
            ("oxygen saturation", r"(?i)\boxygen saturation\s+(?:was\s+)?(\d{2,3}\s*percent)"),
            ("hemoglobin A1c", r"(?i)\bhemoglobin A1c\s+(?:was\s+)?(\d+(?:\.\d+)?\s*percent)"),
            ("LDL cholesterol", r"(?i)\bLDL cholesterol\s+(?:was\s+)?(\d+(?:\.\d+)?\s*mg/dL)"),
            ("creatinine", r"(?i)\bcreatinine\s+(?:was\s+)?(\d+(?:\.\d+)?\s*mg/dL)"),
        ]
        for label, pattern in measurement_patterns:
            mm = re.search(pattern, line)
            if mm:
                dates = [_normalize_date(match.group()) for match in re.finditer(_DATE_PATTERN, line)]
                dates = [value for value in dates if value]
                effective_date = dates[0] if len(set(dates)) == 1 else source.record_date
                _add(facts, source, patient_id, session_id, fact_type="measurement", value=mm.group(1),
                     entity_label=label, quote=quote, start=start, end=end, effective_start=effective_date,
                     normalized=normalize_measurement(label, mm.group(1)))

    # Sentence-level prose extraction avoids duplicating line-level structured facts.
    for start, end, quote in _sentences(body):
        line = " ".join(quote.split())

        # Free-prose problem lists.
        m = re.search(r"(?i)\bsummary\s+lists\s+(.+?)(?:\.|$)", line)
        if not m:
            m = re.search(r"(?i)\b(?:documents?|documented)\s+(.+?)\s+(?:are\s+)?documented\b", line)
        if m and re.search(r"(?i)\b(?:hypertension|diabetes|asthma|rhinitis|hyperlipidemia)\b", m.group(1)):
            for item in [v.strip(" .;") for v in re.split(r"\s+and\s+|\s*;\s*", m.group(1)) if v.strip(" .;")]:
                _add(facts, source, patient_id, session_id, fact_type="condition", value=item,
                     entity_label=item, quote=quote, start=start, end=end)

        # Free-prose medication lists, including two-drug sentences.
        if re.search(r"(?i)\b(?:medication list|are listed|is listed)\b", line):
            prefix = re.sub(r"(?i)\s+(?:are|is)\s+(?:on\s+the\s+medication\s+list|listed).*$", "", line).strip(" .")
            for item in _split_list(prefix):
                match = re.search(r"(?i)\b([A-Za-z][A-Za-z-]{2,35})\s+(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?))(.*)$", item)
                if match:
                    value = " ".join(part.strip() for part in match.groups() if part and part.strip()).strip(" .")
                    _add(facts, source, patient_id, session_id, fact_type="medication", value=value,
                         entity_label=match.group(1), quote=quote, start=start, end=end,
                         normalized=normalize_medication(item, status="current"))

        # Explicit medication lifecycle state. These are separate from the
        # medication-event facts below so current-status answers can consider
        # discontinuation/restart chronology without treating a prescription as
        # proof that a medicine is still active.
        lifecycle_patterns = [
            ("discontinued", "discontinued", r"(?i)\b([A-Za-z][A-Za-z-]{2,35})\s+(?:was\s+)?(?:discontinued|stopped|ceased)\s+(?:on\s+)?(" + _DATE_PATTERN + r")"),
            ("restarted", "current", r"(?i)\b([A-Za-z][A-Za-z-]{2,35})\s+(?:was\s+)?(?:restarted|resumed)\s+(?:on\s+)?(" + _DATE_PATTERN + r")"),
        ]
        for lifecycle_event, lifecycle_status, pattern in lifecycle_patterns:
            lm = re.search(pattern, line)
            if lm and not re.search(r"(?i)\b(?:not|never|possibly|maybe|uncertain)\b", line):
                medication = lm.group(1).casefold()
                lifecycle_date = _normalize_date(lm.group(2))
                if lifecycle_date:
                    _add(
                        facts, source, patient_id, session_id, fact_type="medication",
                        value=medication, entity_label=medication, quote=quote, start=start, end=end,
                        status=lifecycle_status,
                        normalized={**normalize_medication(medication, status=lifecycle_status), "lifecycle_event": lifecycle_event},
                        effective_start=lifecycle_date if lifecycle_status == "current" else None,
                        effective_end=lifecycle_date if lifecycle_status == "discontinued" else None,
                    )

        # Allergy prose. Structured "Medication allergy:" lines were already handled above.
        if re.match(r"(?i)^Medication allerg(?:y|ies)\s*:", line):
            pass
        elif re.search(r"(?i)\bno\s+medication\s+allerg(?:y|ies)\s+is\s+documented\b", line):
            _add(facts, source, patient_id, session_id, fact_type="allergy", value="none documented",
                 entity_label="medication allergy", quote=quote, start=start, end=end,
                 polarity="none_documented", normalized=normalize_allergy("none documented"))
        elif not re.match(r"(?i)^Medication allerg(?:y|ies)\s*:", line):
            m = re.search(r"(?i)\b([A-Za-z][A-Za-z -]{1,60}\s+allergy(?:\s+with\s+[^.;]+)?|[A-Za-z][A-Za-z -]{1,60}\s+caused\s+(?:a\s+)?(?:rash|hives))\b", line)
            if m and re.search(r"(?i)\b(?:allergy|rash|hives)\b", m.group(1)):
                value = re.sub(r"(?i)\s+is\s+documented$", "", m.group(1).strip()).strip()
                _add(facts, source, patient_id, session_id, fact_type="allergy", value=value,
                     entity_label=value, quote=quote, start=start, end=end,
                     normalized=normalize_allergy(value))

        # Pending items: require the item itself to be adjacent to the pending/receipt wording.
        pending_patterns = [
            r"(?i)(?:an?\s+|the\s+)?(.{2,100}?\b(?:report|result|test|testing|study|exam(?:ination)?|records?|consult(?:ation)?|referral))\s+(?:was\s+requested\s+and\s+)?(?:remains?\s+pending|is\s+still\s+(?:pending|outstanding)|has\s+not\s+yet\s+been\s+(?:received|completed)|is\s+pending)",
            r"(?i)(?:requested|pending)\s+(.{2,100}?\b(?:report|result|test|testing|study|exam(?:ination)?|records?|consult(?:ation)?|referral))\s+has\s+not\s+yet\s+been\s+(?:received|completed)",
        ]
        for pattern in pending_patterns:
            pm = re.search(pattern, line)
            if pm:
                item = " ".join(pm.group(1).split()).strip(" .;:")
                _add(facts, source, patient_id, session_id, fact_type="pending_item", value=item,
                     entity_label=item, quote=quote, start=start, end=end,
                     normalized=normalize_pending(item), status="pending")
                break


    # Explicit corrections supersede a prior value/date in the same semantic
    # assertion group. They are indexed before the generic event parser so the
    # conflict resolver can prefer the correction while still retaining the old
    # source fact for audit/provenance.
    for start, end, quote in _sentences(body):
        correction = _correction_medication_event(" ".join(quote.split()))
        if correction:
            medication = correction["medication"]
            event_type = correction["event_type"]
            normalized = {
                "medication": medication,
                "event_type": event_type,
                "date": correction["event_date"],
                "is_correction": True,
            }
            _add(
                facts, source, patient_id, session_id, fact_type="medication_event",
                value=f"{medication} {event_type} {correction['event_date']}",
                entity_label=medication, event_type=event_type, event_date=correction["event_date"],
                quote=quote, start=start, end=end, normalized=normalized,
                supersedes=[{"event_date": correction["supersedes_date"]}],
            )

    # Medication events reuse the already-tested explicit relationship parser.
    known = _known_medications([source])
    for medication in known:
        for event_type in EVENT_TYPES:
            hits, uncertain = _event_mentions(source, medication, event_type, known)
            if uncertain:
                continue
            for hit in hits:
                _add(facts, source, patient_id, session_id, fact_type="medication_event",
                     value=f"{medication} {event_type} {hit.event_date}", entity_label=medication,
                     event_type=event_type, event_date=hit.event_date, quote=hit.quote,
                     start=hit.start, end=hit.end,
                     effective_start=hit.event_date if event_type in {"started", "restarted"} else None,
                     effective_end=hit.event_date if event_type == "discontinued" else None,
                     normalized={"medication": medication, "event_type": event_type, "date": hit.event_date})

    # De-duplicate equivalent assertions while preferring explicit corrections
    # and richer lifecycle metadata. Duplicate source wording can otherwise
    # create artificial conflicts.
    best: dict[tuple, Fact] = {}
    for fact in facts:
        key = (
            fact.fact_type, fact.entity_key, fact.event_type, fact.event_role,
            fact.event_date, fact.value_text, fact.status, fact.polarity, fact.certainty,
        )
        current = best.get(key)
        score = int(bool(fact.normalized.get("is_correction"))) * 10 + len(fact.supersedes) * 5 + int(bool(fact.effective_start or fact.effective_end))
        current_score = -1 if current is None else int(bool(current.normalized.get("is_correction"))) * 10 + len(current.supersedes) * 5 + int(bool(current.effective_start or current.effective_end))
        if current is None or score > current_score:
            best[key] = fact
    return list(best.values())

def _current_source_row(db: sqlite3.Connection, source_kind: str, source_id: str):
    if source_kind == "document":
        return db.execute(
            "SELECT id,patient_id,session_id,filename AS title,text AS body,version,state,context_enabled,updated_at FROM documents WHERE id=?",
            (source_id,),
        ).fetchone()
    if source_kind == "note":
        return db.execute(
            "SELECT id,patient_id,session_id,title,body,version,NULL AS state,context_enabled,updated_at FROM notes WHERE id=?",
            (source_id,),
        ).fetchone()
    raise ValueError("Unsupported source kind")


def _row_to_source(row, kind: str) -> EligibleSource:
    body = row["body"] or ""
    return EligibleSource(
        id=row["id"], kind="R" if kind == "document" else "N", title=row["title"],
        body=body, version=int(row["version"]), record_date=_source_record_date(body),
        updated_at=row["updated_at"],
    )


def sync_source_facts(db: sqlite3.Connection, source_kind: str, source_id: str, *, refresh_resolutions: bool = True) -> dict[str, Any]:
    """Synchronize one derived source index. Safe to call after every source mutation."""
    row = _current_source_row(db, source_kind, source_id)
    previous = db.execute(
        "SELECT patient_id,fact_count FROM clinical_fact_sources WHERE source_kind=? AND source_id=?",
        (source_kind, source_id),
    ).fetchone()
    db.execute("DELETE FROM clinical_facts WHERE source_kind=? AND source_id=?", (source_kind, source_id))
    db.execute("DELETE FROM clinical_fact_sources WHERE source_kind=? AND source_id=?", (source_kind, source_id))
    if row is None:
        previous_patient_id = previous["patient_id"] if previous else None
        if refresh_resolutions and previous_patient_id:
            rebuild_patient_resolutions(db, previous_patient_id)
        return {"indexed": False, "fact_count": 0, "reason": "source_missing", "previous_patient_id": previous_patient_id}
    eligible = bool(row["patient_id"] and row["context_enabled"] and (source_kind == "note" or row["state"] == "approved"))
    if not eligible:
        if refresh_resolutions and row["patient_id"]:
            rebuild_patient_resolutions(db, row["patient_id"])
        return {"indexed": False, "fact_count": 0, "reason": "source_ineligible", "patient_id": row["patient_id"]}

    source = _row_to_source(row, source_kind)
    source_hash = _sha(source.body)
    facts = extract_facts(source, row["patient_id"], row["session_id"])
    indexed_at = now()
    for fact in facts:
        db.execute(
            """INSERT INTO clinical_facts(
               id,patient_id,session_id,fact_type,entity_key,entity_label,event_type,event_role,value_text,
               normalized_json,event_date,asserted_at,effective_start,effective_end,assertion_key,supersedes_json,
               status,polarity,certainty,source_kind,source_id,source_version,source_sha256,source_title,quote,
               start_offset,end_offset,extraction_method,extraction_version,semantic_version,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fact.id, fact.patient_id, fact.session_id, fact.fact_type, fact.entity_key,
                fact.entity_label, fact.event_type, fact.event_role, fact.value_text,
                json.dumps(fact.normalized, sort_keys=True, separators=(",", ":")), fact.event_date,
                fact.asserted_at, fact.effective_start, fact.effective_end, fact.assertion_key,
                json.dumps(fact.supersedes, sort_keys=True, separators=(",", ":")),
                fact.status, fact.polarity, fact.certainty, fact.source_kind, fact.source_id,
                fact.source_version, fact.source_sha256, fact.source_title, fact.quote,
                fact.start_offset, fact.end_offset, fact.extraction_method, fact.extraction_version,
                fact.semantic_version, indexed_at,
            ),
        )
    db.execute(
        """INSERT INTO clinical_fact_sources(
           source_kind,source_id,patient_id,source_version,source_sha256,extraction_version,fact_count,indexed_at
        ) VALUES(?,?,?,?,?,?,?,?)""",
        (source_kind, source_id, row["patient_id"], row["version"], source_hash, EXTRACTION_VERSION, len(facts), indexed_at),
    )
    if refresh_resolutions:
        rebuild_patient_resolutions(db, row["patient_id"])
        if previous and previous["patient_id"] != row["patient_id"]:
            rebuild_patient_resolutions(db, previous["patient_id"])
    return {"indexed": True, "fact_count": len(facts), "patient_id": row["patient_id"], "source_version": row["version"]}


def sync_patient_facts(db: sqlite3.Connection, patient_id: str) -> dict[str, Any]:
    """Bring the derived fact index in sync with every eligible source for a patient."""
    current: dict[tuple[str, str], tuple[int, str]] = {}
    for row in db.execute("SELECT id,text,version FROM documents WHERE patient_id=? AND state='approved' AND context_enabled=1", (patient_id,)):
        current[("document", row["id"])] = (int(row["version"]), _sha(row["text"] or ""))
    for row in db.execute("SELECT id,body,version FROM notes WHERE patient_id=? AND context_enabled=1", (patient_id,)):
        current[("note", row["id"])] = (int(row["version"]), _sha(row["body"] or ""))

    indexed = {
        (row["source_kind"], row["source_id"]): (int(row["source_version"]), row["source_sha256"], row["extraction_version"])
        for row in db.execute("SELECT * FROM clinical_fact_sources WHERE patient_id=?", (patient_id,))
    }
    rebuilt = 0
    removed = 0
    for key in list(indexed):
        if key not in current:
            db.execute("DELETE FROM clinical_facts WHERE source_kind=? AND source_id=?", key)
            db.execute("DELETE FROM clinical_fact_sources WHERE source_kind=? AND source_id=?", key)
            removed += 1
    for key, (version, digest) in current.items():
        existing = indexed.get(key)
        if existing != (version, digest, EXTRACTION_VERSION):
            sync_source_facts(db, key[0], key[1], refresh_resolutions=False)
            rebuilt += 1
    rebuild_patient_resolutions(db, patient_id)
    count = db.execute("SELECT COUNT(*) FROM clinical_facts WHERE patient_id=?", (patient_id,)).fetchone()[0]
    resolution_count = db.execute("SELECT COUNT(*) FROM clinical_fact_resolutions WHERE patient_id=?", (patient_id,)).fetchone()[0]
    return {
        "patient_id": patient_id, "sources_rebuilt": rebuilt, "sources_removed": removed,
        "fact_count": count, "resolution_count": resolution_count,
        "extraction_version": EXTRACTION_VERSION, "semantic_version": SEMANTIC_VERSION,
    }


def facts_for_patient(
    db: sqlite3.Connection, patient_id: str, *, fact_type: str | None = None,
    entity: str | None = None, event_type: str | None = None,
) -> list[Fact]:
    clauses = ["patient_id=?"]
    params: list[Any] = [patient_id]
    if fact_type:
        clauses.append("fact_type=?")
        params.append(fact_type)
    if entity:
        clauses.append("entity_key=?")
        params.append(_key(entity))
    if event_type:
        clauses.append("event_type=?")
        params.append(event_type)
    rows = db.execute(
        "SELECT * FROM clinical_facts WHERE " + " AND ".join(clauses) + " ORDER BY COALESCE(event_date,''),created_at,rowid",
        params,
    ).fetchall()
    return [_fact_from_row(row) for row in rows]


def medication_names(db: sqlite3.Connection, patient_id: str) -> tuple[str, ...]:
    rows = db.execute(
        """SELECT DISTINCT entity_label FROM clinical_facts
           WHERE patient_id=? AND fact_type IN ('medication','medication_event')
             AND entity_label IS NOT NULL ORDER BY entity_label COLLATE NOCASE""",
        (patient_id,),
    ).fetchall()
    return tuple(row[0].casefold() for row in rows if row[0])


def medication_event_facts(db: sqlite3.Connection, patient_id: str, medications: Iterable[str], event_type: str) -> list[Fact]:
    names = {_key(name) for name in medications}
    if not names:
        return []
    placeholders = ",".join("?" for _ in names)
    rows = db.execute(
        f"""SELECT * FROM clinical_facts
            WHERE patient_id=? AND fact_type='medication_event' AND event_type=?
              AND entity_key IN ({placeholders})
            ORDER BY event_date,source_title,start_offset""",
        (patient_id, event_type, *sorted(names)),
    ).fetchall()
    return [_fact_from_row(row) for row in rows]


def _fact_from_row(row) -> Fact:
    keys = set(row.keys())
    return Fact(
        id=row["id"], patient_id=row["patient_id"], session_id=row["session_id"],
        fact_type=row["fact_type"], entity_key=row["entity_key"], entity_label=row["entity_label"],
        event_type=row["event_type"], event_role=row["event_role"] if "event_role" in keys else "unspecified",
        value_text=row["value_text"], normalized=json.loads(row["normalized_json"] or "{}"),
        event_date=row["event_date"], asserted_at=row["asserted_at"] if "asserted_at" in keys else None,
        effective_start=row["effective_start"] if "effective_start" in keys else None,
        effective_end=row["effective_end"] if "effective_end" in keys else None,
        assertion_key=row["assertion_key"] if "assertion_key" in keys else "",
        supersedes=json.loads(row["supersedes_json"] or "[]") if "supersedes_json" in keys else [],
        status=row["status"], polarity=row["polarity"], certainty=row["certainty"],
        source_kind=row["source_kind"], source_id=row["source_id"], source_version=int(row["source_version"]),
        source_sha256=row["source_sha256"], source_title=row["source_title"], quote=row["quote"],
        start_offset=int(row["start_offset"]), end_offset=int(row["end_offset"]),
        extraction_method=row["extraction_method"], extraction_version=row["extraction_version"],
        semantic_version=row["semantic_version"] if "semantic_version" in keys else SEMANTIC_VERSION,
    )


@dataclass(frozen=True)
class FactResolution:
    id: str
    patient_id: str
    resolution_key: str
    fact_type: str
    entity_key: str | None
    event_type: str | None
    event_role: str
    status: str
    preferred_fact_id: str | None
    fact_ids: list[str]
    superseded_fact_ids: list[str]
    values: dict[str, Any]
    computed_at: str

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "resolution_key": self.resolution_key,
            "fact_type": self.fact_type,
            "entity_key": self.entity_key,
            "event_type": self.event_type,
            "event_role": self.event_role,
            "status": self.status,
            "preferred_fact_id": self.preferred_fact_id,
            "fact_ids": list(self.fact_ids),
            "superseded_fact_ids": list(self.superseded_fact_ids),
            "values": dict(self.values),
            "computed_at": self.computed_at,
        }


def _resolution_from_row(row) -> FactResolution:
    return FactResolution(
        id=row["id"], patient_id=row["patient_id"], resolution_key=row["resolution_key"],
        fact_type=row["fact_type"], entity_key=row["entity_key"], event_type=row["event_type"],
        event_role=row["event_role"], status=row["status"], preferred_fact_id=row["preferred_fact_id"],
        fact_ids=json.loads(row["fact_ids_json"] or "[]"),
        superseded_fact_ids=json.loads(row["superseded_fact_ids_json"] or "[]"),
        values=json.loads(row["values_json"] or "{}"), computed_at=row["computed_at"],
    )


def _write_resolution(
    db: sqlite3.Connection, patient_id: str, resolution_key: str, *, fact_type: str,
    entity_key: str | None, event_type: str | None = None, event_role: str = "unspecified",
    status: str, preferred_fact_id: str | None, fact_ids: list[str],
    superseded_fact_ids: list[str] | None = None, values: dict[str, Any] | None = None,
) -> None:
    db.execute(
        """INSERT INTO clinical_fact_resolutions(
             id,patient_id,resolution_key,fact_type,entity_key,event_type,event_role,status,
             preferred_fact_id,fact_ids_json,superseded_fact_ids_json,values_json,computed_at
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            uid(), patient_id, resolution_key, fact_type, entity_key, event_type, event_role, status,
            preferred_fact_id, json.dumps(fact_ids, separators=(",", ":")),
            json.dumps(superseded_fact_ids or [], separators=(",", ":")),
            json.dumps(values or {}, sort_keys=True, separators=(",", ":")), now(),
        ),
    )


_REPEATABLE_MEDICATION_EVENTS = {"renewed", "dispensed", "picked_up", "started", "restarted", "discontinued"}


def _event_resolution_key(fact: Fact) -> str:
    order_id = str(fact.normalized.get("order_id") or "").casefold().strip()
    role = fact.event_role or "unspecified"
    if order_id:
        suffix = "order:" + order_id
    elif fact.event_type in _REPEATABLE_MEDICATION_EVENTS:
        # Repeated renewals/dispenses/lifecycle events are a time series, not a
        # contradiction merely because they occur on different dates. A
        # correction groups with the date it explicitly supersedes.
        superseded_dates = [
            _normalize_date(item.get("event_date"))
            for item in fact.supersedes if isinstance(item, dict) and item.get("event_date")
        ]
        anchor = next((value for value in superseded_dates if value), None) or fact.event_date or "undated"
        suffix = "event:" + str(anchor)
    else:
        # An undifferentiated prescribed date without an order ID cannot safely
        # be assumed to be a separate order. Multiple dates in this group remain
        # conflict/ambiguity evidence rather than "latest wins".
        suffix = ""
    return "|".join([
        "medication_event", fact.entity_key or "", fact.event_type or "",
        role, suffix,
    ])


def _superseded_event_ids(group: list[Fact]) -> set[str]:
    superseded: set[str] = set()
    for correction in group:
        if not correction.supersedes:
            continue
        for rule in correction.supersedes:
            old_date = _normalize_date(rule.get("event_date")) if isinstance(rule, dict) else None
            if not old_date:
                continue
            for fact in group:
                if fact.id != correction.id and fact.event_date == old_date:
                    superseded.add(fact.id)
    return superseded


def _superseded_measurement_ids(group: list[Fact]) -> set[str]:
    superseded: set[str] = set()
    for correction in group:
        for rule in correction.supersedes:
            if not isinstance(rule, dict):
                continue
            old_value = " ".join(str(rule.get("value") or "").split()).casefold()
            old_date = _normalize_date(rule.get("effective_start"))
            for fact in group:
                if fact.id == correction.id:
                    continue
                if old_value and " ".join(fact.value_text.split()).casefold() != old_value:
                    continue
                fact_date = fact.effective_start or fact.asserted_at
                if old_date and fact_date != old_date:
                    continue
                superseded.add(fact.id)
    return superseded


def _medication_state_resolution(
    medication_facts: list[Fact], event_facts: list[Fact]
) -> tuple[str, str | None, list[str], dict[str, Any]]:
    """Resolve current/historical/discontinued state without guessing chronology.

    A dated discontinuation/restart can outrank an older dated assertion. Undated
    current-vs-discontinued evidence is treated as a conflict rather than using
    source update time as a clinical date. Historical/documented mentions alone
    do not make a medication current.
    """
    assertions: list[tuple[str | None, str, Fact]] = []
    for fact in medication_facts:
        state = fact.status if fact.status in {"current", "historical", "documented", "discontinued"} else "documented"
        date_value = fact.effective_end or fact.effective_start or fact.asserted_at
        assertions.append((date_value, state, fact))
    for fact in event_facts:
        if fact.event_type == "discontinued":
            assertions.append((fact.event_date, "discontinued", fact))
        elif fact.event_type in {"restarted", "started"}:
            assertions.append((fact.event_date, "current", fact))
    if not assertions:
        return "unresolved", None, [], {"resolved_state": "unknown", "assertions": []}

    dated = [item for item in assertions if item[0]]
    undated_states = {state for date_value, state, _ in assertions if not date_value}
    preferred: Fact | None = None
    resolved_state = "unknown"
    status = "consistent"
    as_of = None
    if dated:
        as_of = max(str(item[0]) for item in dated)
        latest = [item for item in dated if str(item[0]) == as_of]
        latest_states = {state for _, state, _ in latest if state in {"current", "discontinued"}}
        if len(latest_states) > 1:
            status = "conflict"
            resolved_state = "conflict"
        elif latest_states:
            resolved_state = next(iter(latest_states))
            preferred = next(fact for _, state, fact in latest if state == resolved_state)
        else:
            # A dated historical/documented mention is informative but does not
            # itself establish current state.
            states = {state for _, state, _ in latest}
            resolved_state = "historical" if "historical" in states else "documented"
            preferred = latest[0][2]
        opposing = "discontinued" if resolved_state == "current" else "current"
        if opposing in undated_states:
            status = "conflict"
            resolved_state = "conflict"
            preferred = None
    else:
        strong = {state for _, state, _ in assertions if state in {"current", "discontinued"}}
        if strong == {"current", "discontinued"}:
            status = "conflict"
            resolved_state = "conflict"
        elif "current" in strong:
            resolved_state = "current"
            preferred = next(fact for _, state, fact in assertions if state == "current")
        elif "discontinued" in strong:
            resolved_state = "discontinued"
            preferred = next(fact for _, state, fact in assertions if state == "discontinued")
        elif any(state == "historical" for _, state, _ in assertions):
            resolved_state = "historical"
            preferred = next(fact for _, state, fact in assertions if state == "historical")
        else:
            resolved_state = "documented"
            preferred = assertions[0][2]

    values = {
        "resolved_state": resolved_state,
        "as_of": as_of,
        "assertions": [
            {"fact_id": fact.id, "date": date_value, "state": state}
            for date_value, state, fact in assertions
        ],
    }
    return status, preferred.id if preferred else None, [fact.id for _, _, fact in assertions], values


def rebuild_patient_resolutions(db: sqlite3.Connection, patient_id: str) -> int:
    """Recompute derived conflict/lifecycle clusters for one patient."""
    db.execute("DELETE FROM clinical_fact_resolutions WHERE patient_id=?", (patient_id,))
    rows = db.execute(
        "SELECT * FROM clinical_facts WHERE patient_id=? ORDER BY rowid", (patient_id,)
    ).fetchall()
    facts = [_fact_from_row(row) for row in rows]
    computed = 0

    # Medication-event conflicts and corrections. Original prescriptions and
    # renewals remain separate semantic groups.
    event_groups: dict[str, list[Fact]] = {}
    for fact in facts:
        if fact.fact_type == "medication_event":
            event_groups.setdefault(_event_resolution_key(fact), []).append(fact)
    for key, group in event_groups.items():
        superseded = _superseded_event_ids(group)
        active = [fact for fact in group if fact.id not in superseded]
        values = sorted({fact.event_date for fact in active if fact.event_date})
        if not active:
            status = "superseded"
            preferred = None
        elif len(values) <= 1:
            status = "single" if len(active) == 1 else "consistent"
            corrections = [fact for fact in active if fact.normalized.get("is_correction")]
            preferred = (corrections[0] if corrections else active[0]).id
        else:
            status = "conflict"
            preferred = None
        sample = group[0]
        _write_resolution(
            db, patient_id, key, fact_type="medication_event", entity_key=sample.entity_key,
            event_type=sample.event_type, event_role=sample.event_role, status=status,
            preferred_fact_id=preferred, fact_ids=[fact.id for fact in group],
            superseded_fact_ids=sorted(superseded),
            values={"event_dates": values, "order_id": sample.normalized.get("order_id")},
        )
        computed += 1

    # Medication lifecycle resolution.
    medication_keys = sorted({fact.entity_key for fact in facts if fact.entity_key and fact.fact_type in {"medication", "medication_event"}})
    for entity_key in medication_keys:
        medication_facts = [fact for fact in facts if fact.fact_type == "medication" and fact.entity_key == entity_key]
        lifecycle_events = [
            fact for fact in facts
            if fact.fact_type == "medication_event" and fact.entity_key == entity_key
            and fact.event_type in {"started", "restarted", "discontinued"}
        ]
        if not medication_facts and not lifecycle_events:
            continue
        status, preferred, fact_ids, values = _medication_state_resolution(medication_facts, lifecycle_events)
        _write_resolution(
            db, patient_id, f"medication_state|{entity_key}", fact_type="medication",
            entity_key=entity_key, status=status, preferred_fact_id=preferred,
            fact_ids=fact_ids, values=values,
        )
        computed += 1

    # Measurement conflicts/corrections are scoped to the same measurement and
    # clinical/effective date. Different dates form a time series, not a conflict.
    measurement_groups: dict[str, list[Fact]] = {}
    for fact in facts:
        if fact.fact_type != "measurement":
            continue
        date_key = fact.effective_start or fact.asserted_at or "undated"
        key = f"measurement|{fact.entity_key or ''}|{date_key}"
        measurement_groups.setdefault(key, []).append(fact)
    for key, group in measurement_groups.items():
        superseded = _superseded_measurement_ids(group)
        active = [fact for fact in group if fact.id not in superseded]
        values = sorted({fact.value_text for fact in active})
        status = "superseded" if not active else ("conflict" if len(values) > 1 else ("single" if len(active) == 1 else "consistent"))
        preferred = None
        if active and len(values) == 1:
            corrections = [fact for fact in active if fact.normalized.get("is_correction")]
            preferred = (corrections[0] if corrections else active[0]).id
        sample = group[0]
        _write_resolution(
            db, patient_id, key, fact_type="measurement", entity_key=sample.entity_key,
            status=status, preferred_fact_id=preferred, fact_ids=[fact.id for fact in group],
            superseded_fact_ids=sorted(superseded),
            values={"values": values, "effective_date": sample.effective_start or sample.asserted_at},
        )
        computed += 1
    return computed


def fact_resolutions_for_patient(
    db: sqlite3.Connection, patient_id: str, *, fact_type: str | None = None,
    entity: str | None = None, event_type: str | None = None, status: str | None = None,
) -> list[FactResolution]:
    sync_patient_facts(db, patient_id)
    clauses = ["patient_id=?"]
    params: list[Any] = [patient_id]
    if fact_type:
        clauses.append("fact_type=?"); params.append(fact_type)
    if entity:
        clauses.append("entity_key=?"); params.append(_key(entity))
    if event_type:
        clauses.append("event_type=?"); params.append(event_type)
    if status:
        clauses.append("status=?"); params.append(status)
    rows = db.execute(
        "SELECT * FROM clinical_fact_resolutions WHERE " + " AND ".join(clauses) + " ORDER BY resolution_key",
        params,
    ).fetchall()
    return [_resolution_from_row(row) for row in rows]


def validate_resolution_provenance(
    db: sqlite3.Connection, resolutions: Iterable[FactResolution]
) -> bool:
    """Ensure every fact referenced by a resolution still has exact provenance."""
    resolution_list = list(resolutions)
    expected_ids = {
        fact_id for resolution in resolution_list for fact_id in resolution.fact_ids
        if fact_id
    }
    if not expected_ids:
        return True
    facts = _facts_by_ids(db, sorted(expected_ids))
    if {fact.id for fact in facts} != expected_ids:
        return False
    return validate_fact_provenance(db, facts)


def medication_event_resolution(
    db: sqlite3.Connection, patient_id: str, medication: str, event_type: str,
    *, event_role: str | None = None,
) -> list[FactResolution]:
    sync_patient_facts(db, patient_id)
    clauses = [
        "patient_id=?", "fact_type='medication_event'", "entity_key=?", "event_type=?",
    ]
    params: list[Any] = [patient_id, _key(medication), event_type]
    if event_role:
        clauses.append("event_role=?"); params.append(event_role)
    rows = db.execute(
        "SELECT * FROM clinical_fact_resolutions WHERE " + " AND ".join(clauses) + " ORDER BY event_role,resolution_key",
        params,
    ).fetchall()
    return [_resolution_from_row(row) for row in rows]


def medication_state_resolution(
    db: sqlite3.Connection, patient_id: str, medication: str
) -> FactResolution | None:
    sync_patient_facts(db, patient_id)
    row = db.execute(
        "SELECT * FROM clinical_fact_resolutions WHERE patient_id=? AND resolution_key=?",
        (patient_id, f"medication_state|{_key(medication)}"),
    ).fetchone()
    return _resolution_from_row(row) if row else None


def validate_fact_provenance(db: sqlite3.Connection, facts: Iterable[Fact]) -> bool:
    """Recheck current source version/hash and exact quote offsets for every fact."""
    for fact in facts:
        row = _current_source_row(db, fact.source_kind, fact.source_id)
        if row is None or row["patient_id"] != fact.patient_id or int(row["version"]) != fact.source_version:
            return False
        eligible = bool(row["context_enabled"] and (fact.source_kind == "note" or row["state"] == "approved"))
        if not eligible or _sha(row["body"] or "") != fact.source_sha256:
            return False
        body = row["body"] or ""
        if fact.start_offset < 0 or fact.end_offset > len(body) or body[fact.start_offset:fact.end_offset] != fact.quote:
            return False
    return True


def fact_inventory(db: sqlite3.Connection, patient_id: str) -> dict[str, Any]:
    sync = sync_patient_facts(db, patient_id)
    rows = db.execute(
        "SELECT fact_type,COUNT(*) AS count FROM clinical_facts WHERE patient_id=? GROUP BY fact_type ORDER BY fact_type",
        (patient_id,),
    ).fetchall()
    resolution_rows = db.execute(
        "SELECT status,COUNT(*) AS count FROM clinical_fact_resolutions WHERE patient_id=? GROUP BY status ORDER BY status",
        (patient_id,),
    ).fetchall()
    return {
        **sync,
        "by_type": {row["fact_type"]: row["count"] for row in rows},
        "resolutions_by_status": {row["status"]: row["count"] for row in resolution_rows},
    }



def _definite(value: str) -> str:
    clean = value.strip(" .")
    return clean if re.match(r"(?i)^(?:the|a|an)\b", clean) else "the " + clean

def _join(values: list[str]) -> str:
    values = [v.strip(" .") for v in values if v and v.strip(" .")]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _best_per_entity(facts: Iterable[Fact]) -> list[Fact]:
    best: dict[str, Fact] = {}
    for fact in facts:
        key = fact.entity_key or _key(fact.value_text)
        current = best.get(key)
        if current is None:
            best[key] = fact
        elif fact.fact_type != "pending_item" and len(fact.value_text) > len(current.value_text):
            best[key] = fact
    return list(best.values())


def _facts_by_ids(db: sqlite3.Connection, fact_ids: Iterable[str]) -> list[Fact]:
    ids = [fact_id for fact_id in fact_ids if fact_id]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = db.execute(
        f"SELECT * FROM clinical_facts WHERE id IN ({placeholders})", ids
    ).fetchall()
    by_id = {row["id"]: _fact_from_row(row) for row in rows}
    return [by_id[fact_id] for fact_id in ids if fact_id in by_id]


def _medication_state_groups(db: sqlite3.Connection, patient_id: str) -> tuple[list[Fact], list[Fact], list[Fact], list[tuple[str, list[Fact]]]]:
    """Return current, historical/documented, discontinued, and conflicting medication evidence."""
    resolutions = db.execute(
        "SELECT * FROM clinical_fact_resolutions WHERE patient_id=? AND fact_type='medication' ORDER BY resolution_key",
        (patient_id,),
    ).fetchall()
    current: list[Fact] = []
    noncurrent: list[Fact] = []
    discontinued: list[Fact] = []
    conflicts: list[tuple[str, list[Fact]]] = []
    for row in resolutions:
        resolution = _resolution_from_row(row)
        evidence = _facts_by_ids(db, resolution.fact_ids)
        label = next((fact.entity_label for fact in evidence if fact.entity_label), resolution.entity_key or "medication")
        state = str(resolution.values.get("resolved_state") or "unknown")
        if resolution.status == "conflict" or state == "conflict":
            conflicts.append((label, evidence))
            continue
        if state == "current":
            as_of = resolution.values.get("as_of")
            candidates = [
                fact for fact in evidence
                if fact.fact_type == "medication" and fact.status == "current"
                and (not as_of or (fact.asserted_at and str(fact.asserted_at) >= str(as_of)))
            ]
            # If a later start/restart established current status, do not reuse an
            # older pre-event dose merely because it appears in an earlier current
            # medication list. Prefer a current-medication assertion at/after the
            # resolved state date; otherwise retain the lifecycle event itself.
            if candidates:
                current.append(max(candidates, key=lambda fact: (str(fact.asserted_at or ""), len(fact.value_text))))
            elif resolution.preferred_fact_id:
                preferred = _facts_by_ids(db, [resolution.preferred_fact_id])
                if preferred:
                    current.append(preferred[0])
        elif state == "discontinued":
            candidates = [fact for fact in evidence if fact.fact_type == "medication" and fact.status == "discontinued"]
            if candidates:
                discontinued.append(max(candidates, key=lambda fact: len(fact.value_text)))
            elif resolution.preferred_fact_id:
                preferred = _facts_by_ids(db, [resolution.preferred_fact_id])
                if preferred:
                    discontinued.append(preferred[0])
        else:
            candidates = [fact for fact in evidence if fact.fact_type == "medication" and fact.status in {"historical", "documented"}]
            if candidates:
                noncurrent.append(max(candidates, key=lambda fact: len(fact.value_text)))
    return current, noncurrent, discontinued, conflicts


def _measurement_latest_groups(db: sqlite3.Connection, patient_id: str) -> tuple[list[Fact], list[tuple[str, list[Fact], str | None]]]:
    rows = db.execute(
        "SELECT * FROM clinical_fact_resolutions WHERE patient_id=? AND fact_type='measurement' ORDER BY resolution_key",
        (patient_id,),
    ).fetchall()
    by_entity: dict[str, list[FactResolution]] = {}
    for row in rows:
        resolution = _resolution_from_row(row)
        by_entity.setdefault(resolution.entity_key or "measurement", []).append(resolution)
    resolved: list[Fact] = []
    conflicts: list[tuple[str, list[Fact], str | None]] = []
    for entity_key, resolutions in by_entity.items():
        def sort_key(resolution: FactResolution):
            return str(resolution.values.get("effective_date") or "")
        latest = sorted(resolutions, key=sort_key, reverse=True)[0]
        evidence = _facts_by_ids(db, latest.fact_ids)
        label = next((fact.entity_label for fact in evidence if fact.entity_label), entity_key)
        effective_date = latest.values.get("effective_date")
        if latest.status == "conflict":
            conflicts.append((label, evidence, effective_date))
        elif latest.preferred_fact_id:
            preferred = _facts_by_ids(db, [latest.preferred_fact_id])
            if preferred:
                resolved.append(preferred[0])
    return resolved, conflicts


def _medication_current_display(fact: Fact) -> str:
    lifecycle = fact.normalized.get("lifecycle_event") if isinstance(fact.normalized, dict) else None
    if (
        (fact.fact_type == "medication_event" and fact.event_type in {"started", "restarted"})
        or lifecycle in {"started", "restarted"}
    ):
        label = fact.entity_label or fact.normalized.get("medication") or "medication"
        event_word = "restarted" if (fact.event_type == "restarted" or lifecycle == "restarted") else "started"
        event_date = fact.event_date or fact.effective_start
        date_text = f" on {event_date}" if event_date else ""
        return f"{label} ({event_word}{date_text}; later current dose not documented)"
    return fact.value_text


def answer_from_facts(db: sqlite3.Connection, patient_id: str, intent: str, question: str = "") -> tuple[str, list[Fact]] | None:
    """Build direct structured answers from the persistent provenance index."""
    sync_patient_facts(db, patient_id)
    if intent == "medications":
        current, noncurrent, discontinued, conflicts = _medication_state_groups(db, patient_id)
        if conflicts:
            conflict_labels = [label for label, _ in conflicts]
            used = [fact for _, facts in conflicts for fact in facts][:8] + current[:4]
            message = (
                "The reviewed sources contain conflicting current-status documentation for "
                + _join(conflict_labels) + "."
            )
            if current:
                message += " Other medications currently supported by the reviewed record are " + "; ".join(f.value_text for f in current) + "."
            return message, used
        if current:
            answer = "The current medications are " + "; ".join(_medication_current_display(fact) for fact in current) + "."
            if discontinued:
                answer += " The chart also documents discontinued medication: " + "; ".join(fact.entity_label or fact.value_text for fact in discontinued) + "."
            return answer, current + discontinued[:3]
        if discontinued:
            return "The reviewed chart does not establish a current medication list; it documents discontinued medication: " + "; ".join(fact.entity_label or fact.value_text for fact in discontinued) + ".", discontinued
        if noncurrent:
            return "The reviewed records mention " + "; ".join(fact.value_text for fact in noncurrent) + ", but this fact index does not establish that they are current medications.", noncurrent
        return None
    if intent == "conditions":
        facts = _best_per_entity(facts_for_patient(db, patient_id, fact_type="condition"))
        if not facts:
            return None
        return f"The chart documents {_join([fact.value_text for fact in facts])}.", facts
    if intent == "allergies":
        facts = facts_for_patient(db, patient_id, fact_type="allergy")
        affirmed = _best_per_entity([fact for fact in facts if fact.polarity == "affirmed"])
        absent = [fact for fact in facts if fact.polarity == "none_documented"]
        if affirmed and absent:
            used = affirmed[:3] + absent[:1]
            return (
                "The reviewed sources contain conflicting allergy documentation: "
                + _join([fact.value_text for fact in affirmed[:3]])
                + "; another reviewed source says no medication allergy is documented.",
                used,
            )
        if affirmed:
            return f"The chart documents {_join([fact.value_text for fact in affirmed[:3]])}.", affirmed[:3]
        if absent:
            return "The reviewed chart says no medication allergy is documented.", absent[:1]
        return None
    if intent == "pending_items":
        facts = _best_per_entity(facts_for_patient(db, patient_id, fact_type="pending_item"))
        if not facts:
            return None
        labels = [fact.entity_label or fact.value_text for fact in facts[:4]]
        q = question.casefold()
        if re.search(r"\b(?:nothing\s+(?:is\s+)?missing|so\s+nothing|none\s+missing)\b", q):
            pending_text = _definite(labels[0]) if len(labels) == 1 else _join(labels)
            return f"No. The chart still lists {pending_text} as pending.", facts[:4]
        if len(labels) == 1:
            return f"The chart says that {_definite(labels[0])} is still pending.", facts[:1]
        return f"The chart lists these outstanding items: {_join(labels)}.", facts[:4]
    if intent == "vitals_labs":
        facts, conflicts = _measurement_latest_groups(db, patient_id)
        if not facts and not conflicts:
            return None
        pieces: list[str] = []
        used: list[Fact] = []
        if facts:
            values = [f"{fact.entity_label} {fact.value_text}" for fact in facts[:6]]
            pieces.append(f"The latest resolved reviewed measurements include {_join(values)}.")
            used.extend(facts[:6])
        for label, evidence, effective_date in conflicts[:3]:
            values = sorted({fact.value_text for fact in evidence})
            date_text = f" for {effective_date}" if effective_date else ""
            pieces.append(f"The reviewed sources conflict on {label}{date_text}: {_join(values)}.")
            used.extend(evidence[:4])
        return " ".join(pieces), used
    if intent == "plan_follow_up":
        facts = facts_for_patient(db, patient_id, fact_type="plan")
        unique: list[Fact] = []
        seen: set[str] = set()
        for fact in facts:
            key = re.sub(r"\W+", " ", fact.value_text.casefold()).strip()
            if key not in seen:
                seen.add(key); unique.append(fact)
        if not unique:
            return None
        return f"The documented plan is to {_join([fact.value_text for fact in unique[:3]])}.", unique[:3]
    if intent == "chart_overview":
        pieces: list[str] = []
        used: list[Fact] = []
        conditions = _best_per_entity(facts_for_patient(db, patient_id, fact_type="condition"))[:2]
        medications, documented_medications, discontinued_medications, medication_conflicts = _medication_state_groups(db, patient_id)
        medications = medications[:2]
        documented_medications = documented_medications[:2]
        allergies = _best_per_entity([f for f in facts_for_patient(db, patient_id, fact_type="allergy") if f.polarity == "affirmed"])[:1]
        pending = _best_per_entity(facts_for_patient(db, patient_id, fact_type="pending_item"))[:2]
        if conditions:
            pieces.append(f"The chart documents {_join([f.value_text for f in conditions])}."); used.extend(conditions)
        if medication_conflicts:
            pieces.append("Medication status is conflicting for " + _join([label for label, _ in medication_conflicts[:2]]) + ".")
            used.extend([fact for _, facts in medication_conflicts[:2] for fact in facts][:6])
        elif medications:
            pieces.append("Current medications are " + "; ".join(_medication_current_display(f) for f in medications) + "."); used.extend(medications)
        elif documented_medications:
            pieces.append("Medication mentions are present, but the reviewed facts do not establish a current medication list."); used.extend(documented_medications)
        if allergies:
            pieces.append(f"It notes {_join([f.value_text for f in allergies])}."); used.extend(allergies)
        if pending:
            pieces.append(f"The remaining outstanding items are {_join([f.entity_label or f.value_text for f in pending])}."); used.extend(pending)
        return (" ".join(pieces), used) if pieces else None
    return None


def fact_sources_for_ui(facts: Iterable[Fact], limit: int = 4) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Fact]] = {}
    for fact in facts:
        grouped.setdefault((fact.source_kind, fact.source_id, fact.source_version), []).append(fact)
    output: list[dict[str, Any]] = []
    counts = {"R": 0, "N": 0}
    for (kind, source_id, version), source_facts in list(grouped.items())[:limit]:
        label_kind = "R" if kind == "document" else "N"
        counts[label_kind] += 1
        quotes: list[str] = []
        for fact in source_facts:
            if fact.quote not in quotes:
                quotes.append(fact.quote)
        excerpt = "\n".join(quotes[:3])[:1000]
        first = source_facts[0]
        output.append({
            "label": f"{label_kind}{counts[label_kind]}",
            "id": source_id,
            "title": first.source_title,
            "version": version,
            "excerpt": excerpt,
            "partial": True,
            "score": 100,
            "kind": label_kind,
            "record_date": first.event_date,
            "document_type": None,
            "fact_ids": [fact.id for fact in source_facts],
        })
    return output
