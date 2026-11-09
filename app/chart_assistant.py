"""Deterministic chart-question routing and answer construction.

The local 500M-class model is useful for flexible language generation, but it is
not reliable enough to decide what a medication, pending item, condition, or
latest report is. This module resolves common chart intents, sanitizes and
ranks reviewed evidence, removes duplicate sources, and builds concise factual
answers directly from the record. The model remains available for genuinely
open-ended questions after these record-native paths are exhausted.
"""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import re
from typing import Iterable

INTENT_PENDING = "pending_items"
INTENT_MEDICATIONS = "medications"
INTENT_CONDITIONS = "conditions"
INTENT_LATEST_REPORT = "latest_report"
INTENT_OVERVIEW = "chart_overview"
INTENT_ALLERGIES = "allergies"
INTENT_VITALS_LABS = "vitals_labs"
INTENT_PLAN = "plan_follow_up"
INTENT_OPEN = "open_question"

STOP_WORDS = {
    "a", "about", "all", "an", "and", "are", "as", "at", "available", "be", "been",
    "can", "chart", "could", "did", "do", "does", "for", "from", "give", "has", "have",
    "he", "her", "his", "i", "in", "information", "is", "it", "its", "know", "latest",
    "me", "of", "on", "patient", "please", "record", "records", "report", "reports", "she",
    "so", "summarize", "summary", "tell", "that", "the", "their", "there", "they", "this",
    "to", "using", "was", "were", "what", "which", "who", "with", "you",
}

INTENT_TERMS = {
    INTENT_PENDING: {
        "pending", "missing", "outstanding", "requested", "received", "completed", "needed",
        "awaiting", "unavailable", "follow-up", "report", "result", "test",
    },
    INTENT_MEDICATIONS: {
        "medication", "medications", "medicine", "medicines", "meds", "drug", "drugs",
        "prescribed", "prescription", "taking", "using", "tablet", "inhaler", "spray", "dose",
    },
    INTENT_CONDITIONS: {
        "condition", "conditions", "diagnosis", "diagnoses", "problem", "problems", "disease",
        "health", "status", "hypertension", "diabetes", "asthma",
    },
    INTENT_LATEST_REPORT: {
        "latest", "recent", "newest", "report", "document", "source", "review",
    },
    INTENT_OVERVIEW: {
        "overview", "summary", "documented", "know", "chart", "record", "information",
    },
    INTENT_ALLERGIES: {"allergy", "allergies", "allergic", "reaction", "reactions"},
    INTENT_VITALS_LABS: {
        "vital", "vitals", "blood", "pressure", "pulse", "temperature", "oxygen", "weight",
        "laboratory", "lab", "labs", "a1c", "ldl", "creatinine", "glucose", "result", "results",
    },
    INTENT_PLAN: {"plan", "follow-up", "followup", "next", "return", "monitor", "continue"},
}

DEMO_MARKERS = (
    "synthetic training", "synthetic record", "synthetic test", "demo data", "fictional patient",
    "not a real patient", "for testing only", "software fixture", "training record",
)

_METADATA_RE = re.compile(
    r"^(?:patient(?:\s+name)?|name|dob|date\s+of\s+birth|mrn|medical\s+record\s+(?:number|id)|source|page)\s*:",
    re.I,
)
_REPORT_DATE_RE = re.compile(r"(?im)^\s*(?:report|document)\s+date\s*:\s*(\d{4}-\d{2}-\d{2})\s*$")
_DOCUMENT_TYPE_RE = re.compile(r"(?im)^\s*document\s+type\s*:\s*([^\r\n]+)")


def history_dicts(history: Iterable) -> list[dict]:
    return [turn.model_dump() if hasattr(turn, "model_dump") else dict(turn) for turn in history][-8:]


def last_user_questions(history: list[dict], limit: int = 4) -> list[str]:
    values = [
        str(turn.get("content") or "").strip()
        for turn in history
        if turn.get("role") == "user" and str(turn.get("content") or "").strip()
    ]
    return values[-limit:]


def _direct_intent(question: str) -> str | None:
    q = " ".join(question.casefold().split())
    if re.search(r"\b(?:latest|most recent|newest)\b.*\b(?:report|document|record|note|source)\b", q) or re.search(
        r"\bsummar(?:y|ize)\b.*\b(?:latest|most recent|newest)\b", q
    ):
        return INTENT_LATEST_REPORT
    if re.search(
        r"\b(?:medication|medications|medicine|medicines|meds|drug|drugs|prescription|prescribed)\b", q
    ) or re.search(r"\bwhat\s+(?:is|are)\s+(?:the\s+)?patient\s+(?:taking|using)\b", q):
        return INTENT_MEDICATIONS
    if re.search(r"\b(?:allergy|allergies|allergic|reaction)\b", q):
        return INTENT_ALLERGIES
    if re.search(
        r"\b(?:missing|pending|outstanding|awaiting|still\s+needed|not\s+received|not\s+completed|nothing\s+is\s+missing|nothing\s+missing)\b",
        q,
    ) or re.search(r"\bwhich\b.*\b(?:report|result|test)\b.*\b(?:missing|pending|outstanding)\b", q):
        return INTENT_PENDING
    if re.search(r"\b(?:condition|conditions|diagnosis|diagnoses|active\s+problems?|health\s+status)\b", q):
        return INTENT_CONDITIONS
    if re.search(r"\b(?:vitals?|blood\s+pressure|pulse|temperature|oxygen\s+saturation|laborator(?:y|ies)|labs?|a1c|ldl|creatinine|glucose)\b", q):
        return INTENT_VITALS_LABS
    if re.search(r"\b(?:plan|follow[- ]?up|next\s+steps?|return\s+in|monitoring)\b", q):
        return INTENT_PLAN
    if re.search(
        r"\b(?:what\s+information\s+is\s+documented|what\s+do\s+you\s+know|summar(?:y|ize)\s+(?:the\s+)?(?:patient|chart|record)|chart\s+summary|record\s+summary|overview)\b",
        q,
    ) or re.search(r"\b(?:summarize|summary)\b", q):
        return INTENT_OVERVIEW
    return None


def direct_intent(question: str) -> str | None:
    """Public, deterministic intent classifier used by dialogue-state resolution."""
    return _direct_intent(question)


_AMBIGUOUS_FOLLOW_UP = re.compile(
    r"^\s*(?:so\??|really\??|are\s+you\s+sure\??|what\s+about\s+(?:it|that|them)\??|"
    r"tell\s+me\s+more\??|can\s+you\s+clarify\??|i(?:'m|\s+am)\s+asking\s+(?:about|for)\s+.+|"
    r"and\s+what\s+else\??|what\s+else\??|when\s+was\s+it\s+requested\??|"
    r"when\s+did\s+they\s+request\s+it\??|what\s+dose\??|how\s+often\??|which\s+one\??)\s*$",
    re.I,
)


def resolve_intent(question: str, history: list[dict]) -> str:
    direct = _direct_intent(question)
    if direct:
        return direct
    if _AMBIGUOUS_FOLLOW_UP.match(question):
        for previous in reversed(last_user_questions(history, 4)):
            inherited = _direct_intent(previous)
            if inherited:
                return inherited
    return INTENT_OPEN


def query_terms(question: str, history: list[dict], intent: str, referent: dict | None = None) -> set[str]:
    previous = last_user_questions(history, 2)
    text = " ".join(previous + [question]).casefold()
    if referent:
        selected = referent.get("selected") or {}
        values = [selected.get("label"), selected.get("name"), selected.get("title"), selected.get("value")]
        values.extend(item.get("label") or item.get("name") or item.get("title") or item.get("value") for item in (referent.get("items") or [])[:3] if isinstance(item, dict))
        text += " " + " ".join(str(value) for value in values if value)
    terms = {token for token in re.findall(r"[a-z0-9][a-z0-9-]{2,}", text) if token not in STOP_WORDS}
    terms.update(INTENT_TERMS.get(intent, set()))
    return terms


def source_metadata(text: str) -> tuple[str | None, str | None]:
    date_match = _REPORT_DATE_RE.search(text or "")
    type_match = _DOCUMENT_TYPE_RE.search(text or "")
    return (date_match.group(1) if date_match else None, type_match.group(1).strip() if type_match else None)


def sanitize_record_text(text: str) -> str:
    """Remove demo labels and patient identifiers from assistant-facing excerpts.

    Originals remain unchanged in the encrypted chart. This only reduces noisy
    headers and prevents the tiny model from echoing demo/identifier blocks.
    """
    cleaned: list[str] = []
    for raw in str(text or "").splitlines():
        line = " ".join(raw.strip().split())
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        lower = line.casefold()
        if any(marker in lower for marker in DEMO_MARKERS):
            # A banner-only line disappears. If a test fixture combines its
            # banner and useful content on one line, retain the useful suffix.
            for marker in DEMO_MARKERS:
                line = re.sub(re.escape(marker), "", line, flags=re.I)
            line = line.strip(" —–-:;,. ")
            if not line:
                continue
        if _METADATA_RE.match(line):
            continue
        # Keep report date and document type: they are useful for "latest report".
        cleaned.append(line)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return "\n".join(cleaned).strip()


def _normalized_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9][a-z0-9-]{2,}", text.casefold())
        if token not in STOP_WORDS and token not in {"reviewed", "documented", "recorded"}
    }


def source_similarity(first: str, second: str) -> float:
    a, b = _normalized_tokens(first), _normalized_tokens(second)
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    jaccard = intersection / union
    containment = intersection / min(len(a), len(b))
    return max(jaccard, containment * 0.92)


def deduplicate_sources(sources: list[dict], limit: int) -> list[dict]:
    selected: list[dict] = []
    exact: set[str] = set()
    for source in sources:
        excerpt = str(source.get("excerpt") or "")
        digest = hashlib.sha256(re.sub(r"\W+", " ", excerpt.casefold()).strip().encode()).hexdigest()
        if digest in exact:
            continue
        if any(source_similarity(excerpt, existing["excerpt"]) >= 0.82 for existing in selected):
            continue
        exact.add(digest)
        selected.append(source)
        if len(selected) >= limit:
            break
    return selected


def source_score(text: str, intent: str, terms: set[str]) -> int:
    lower = text.casefold()
    score = sum(min(lower.count(term), 4) for term in terms)
    patterns = {
        INTENT_PENDING: r"\b(?:pending|outstanding|requested|not\s+yet\s+(?:received|completed)|has\s+not\s+been\s+received|remains\s+pending|still\s+outstanding)\b",
        INTENT_MEDICATIONS: r"\b(?:current\s+medications?|medication\s+list|\d+(?:\.\d+)?\s*(?:mg|mcg)|tablet|capsule|inhaler|nasal\s+spray)\b",
        INTENT_CONDITIONS: r"\b(?:active\s+problems?|diagnos(?:is|es)|hypertension|diabetes|asthma|hyperlipidemia|rhinitis)\b",
        INTENT_ALLERGIES: r"\b(?:allerg(?:y|ies|ic)|rash|hives|anaphylaxis)\b",
        INTENT_VITALS_LABS: r"\b(?:blood\s+pressure|pulse|temperature|oxygen\s+saturation|hemoglobin\s+a1c|ldl|creatinine|glucose)\b",
        INTENT_PLAN: r"\b(?:documented\s+plan|plan:|follow[- ]?up|return\s+in|continue\s+the|monitor|obtain\s+the)\b",
    }
    pattern = patterns.get(intent)
    if pattern:
        score += 12 * len(re.findall(pattern, lower, re.I))
    if intent == INTENT_OVERVIEW:
        score += 4 * len(re.findall(r"\b(?:active\s+problems?|current\s+medications?|medication\s+allerg|documented\s+plan)\b", lower))
    return score


def _fact_lines(sources: list[dict]) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()
    for source in sources:
        text = sanitize_record_text(str(source.get("excerpt") or ""))
        for raw in text.splitlines():
            line = " ".join(raw.strip().lstrip("-• ").split())
            if not line:
                continue
            if re.match(r"^(?:report|document)\s+date\s*:", line, re.I) or re.match(r"^document\s+type\s*:", line, re.I):
                continue
            # Preserve short structured lists. For structured narrative lines,
            # retain the label on the first sentence and split later sentences so
            # a pending report is not fused with the whole follow-up plan.
            if re.match(r"^(?:active problems|current medications|medication allerg(?:y|ies))\s*:", line, re.I):
                pieces = [line]
            else:
                pieces = re.split(r"(?<=[.!?])\s+", line)
            for piece in pieces:
                fact = " ".join(piece.strip().split()).strip(" -")
                if len(fact) < 7:
                    continue
                key = re.sub(r"\W+", " ", fact.casefold()).strip()
                if key and key not in seen:
                    seen.add(key)
                    facts.append(fact.rstrip(".") + ".")
    return facts


def _category_facts(sources: list[dict]) -> dict[str, list[str]]:
    categories = {
        INTENT_PENDING: [], INTENT_MEDICATIONS: [], INTENT_CONDITIONS: [], INTENT_ALLERGIES: [],
        INTENT_VITALS_LABS: [], INTENT_PLAN: [], "status": [],
    }
    for fact in _fact_lines(sources):
        lower = fact.casefold()
        if re.search(r"\b(?:pending|outstanding|requested|not\s+yet\s+(?:received|completed)|has\s+not\s+been\s+received|remains\s+pending|still\s+outstanding)\b", lower):
            categories[INTENT_PENDING].append(fact)
        if re.match(r"^current medications\s*:", fact, re.I) or re.search(r"\b(?:\d+(?:\.\d+)?\s*(?:mg|mcg)|tablet|capsule|inhaler|nasal\s+spray)\b", lower) and re.search(r"\b(?:listed|medication|metformin|lisinopril|atorvastatin|cetirizine|albuterol|fluticasone)\b", lower):
            categories[INTENT_MEDICATIONS].append(fact)
        if re.match(r"^active problems\s*:", fact, re.I) or re.search(r"\b(?:hypertension|diabetes|hyperlipidemia|asthma|allergic\s+rhinitis|osteoarthritis|hypothyroidism|migraine|reflux|anxiety)\b", lower):
            categories[INTENT_CONDITIONS].append(fact)
        if (
            re.match(r"^medication allerg(?:y|ies)\s*:", fact, re.I)
            or re.search(r"\b(?:no\s+medication\s+allerg(?:y|ies)|[a-z-]+\s+allergy\s+with|allergy\s+with\s+(?:rash|hives)|caused\s+(?:a\s+)?(?:rash|hives)|anaphylaxis)\b", lower)
        ):
            categories[INTENT_ALLERGIES].append(fact)
        if re.search(r"\b(?:blood\s+pressure|pulse|temperature|oxygen\s+saturation|hemoglobin\s+a1c|ldl|creatinine|glucose|weight)\b", lower):
            categories[INTENT_VITALS_LABS].append(fact)
        if re.match(r"^documented plan\s*:", fact, re.I) or re.search(r"\b(?:follow[- ]?up|return\s+in|continue\s+the|keep\s+a|repeat\s+a1c|obtain\s+the|arrange\s+routine)\b", lower):
            categories[INTENT_PLAN].append(fact)
        if re.search(r"\b(?:denied|no\s+interval|no\s+recent|no\s+new|stable|clear|taking\s+medications\s+consistently|side\s+effects)\b", lower):
            categories["status"].append(fact)
    return categories


def _after_colon(fact: str) -> str:
    value = fact.rstrip(".")
    return value.split(":", 1)[1].strip() if ":" in value else value


def _clean_list_text(value: str) -> str:
    return re.sub(r"\s*;\s*", "; ", value.strip().rstrip("."))


def _pending_item(fact: str) -> str:
    text = fact.strip().rstrip(".")
    patterns = [
        r"^(?:An?\s+|The\s+)?(.+?\breport)\s+(?:was\s+requested\s+and\s+remains\s+pending|has\s+been\s+requested\s+but\s+has\s+not\s+yet\s+been\s+received|has\s+not\s+yet\s+been\s+received|has\s+not\s+been\s+received|remains\s+pending|is\s+(?:still\s+)?pending)$",
        r"^(?:An?\s+|The\s+)?(.+?\bresult)\s+(?:is\s+still\s+outstanding|remains\s+pending|is\s+pending)$",
        r"^(?:An?\s+|The\s+)?(.+?\btesting)\s+(?:has\s+not\s+yet\s+been\s+completed|remains\s+pending|is\s+pending)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, re.I)
        if match:
            item = match.group(1).strip()
            item = re.sub(r"^(?:the\s+)?requested\s+", "", item, flags=re.I)
            return item[0].lower() + item[1:] if item else item
    # Remove surrounding prose when the item cannot be cleanly isolated.
    text = re.sub(r"^(?:Documented plan:\s*)", "", text, flags=re.I)
    text = re.sub(r"^(?:the\s+)?requested\s+", "", text, flags=re.I)
    return text[0].lower() + text[1:] if text else text


def _unique_values(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    keys: list[set[str]] = []
    for value in values:
        clean = " ".join(value.split()).strip(" .;,")
        if not clean:
            continue
        tokens = _normalized_tokens(clean)
        if any(tokens and len(tokens & existing) / min(len(tokens), len(existing)) >= 0.85 for existing in keys if existing):
            continue
        output.append(clean)
        keys.append(tokens)
    return output



def _condition_values(categories: dict[str, list[str]]) -> list[str]:
    structured = [
        _after_colon(fact) for fact in categories[INTENT_CONDITIONS]
        if re.match(r"^active problems\s*:", fact, re.I)
    ]
    if structured:
        return _unique_values(structured)
    values = []
    for fact in categories[INTENT_CONDITIONS]:
        value = fact.rstrip(".")
        value = re.sub(r"^(?:The\s+)?reviewed\s+outside\s+summary\s+lists\s+", "", value, flags=re.I)
        value = re.sub(r"^(?:The\s+)?chart\s+(?:documents|lists)\s+", "", value, flags=re.I)
        value = re.sub(r"\s+(?:are|is)\s+documented$", "", value, flags=re.I)
        values.append(value)
    return _unique_values(values)


def _medication_values(categories: dict[str, list[str]]) -> list[str]:
    structured = [
        _after_colon(fact) for fact in categories[INTENT_MEDICATIONS]
        if re.match(r"^current medications\s*:", fact, re.I)
    ]
    if structured:
        return _unique_values(structured)
    values = []
    for fact in categories[INTENT_MEDICATIONS]:
        value = fact.rstrip(".")
        match = re.match(r"^(.*?)\s+(?:are|is)\s+(?:on|in)\s+the\s+medication\s+list$", value, re.I)
        if not match:
            match = re.match(r"^(.*?)\s+(?:are|is)\s+listed$", value, re.I)
        values.append(match.group(1) if match else value)
    return _unique_values(values)


def _allergy_values(categories: dict[str, list[str]]) -> list[str]:
    values = []
    for fact in categories[INTENT_ALLERGIES]:
        value = _after_colon(fact).rstrip(".")
        value = re.sub(r"\s+(?:is|are)\s+documented$", "", value, flags=re.I)
        values.append(value)
    return _unique_values(values)


def _plan_values(categories: dict[str, list[str]]) -> list[str]:
    return _unique_values(_after_colon(fact) for fact in categories[INTENT_PLAN])


def _vital_values(categories: dict[str, list[str]]) -> list[str]:
    return _unique_values(_after_colon(fact) for fact in categories[INTENT_VITALS_LABS])

def _format_join(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"



def _pending_display(item: str) -> str:
    clean = item.strip()
    if re.match(r"^(?:the|a|an)\s+", clean, re.I):
        return clean
    return "the " + clean

def _format_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = date.fromisoformat(value[:10])
    except ValueError:
        return value
    return parsed.strftime("%B %-d, %Y")


def _split_structured_values(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        for part in re.split(r"\s*;\s*|\s*,\s+(?=[A-Za-z])", value):
            clean = part.strip(" .")
            if clean:
                output.append(clean)
    return _unique_values(output)


def _medication_label(value: str) -> str:
    clean = value.strip()
    match = re.match(
        r"^([A-Za-z][A-Za-z0-9' -]{1,60}?)(?=\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b|\s+(?:tablet|capsule|inhaler|nasal\s+spray)\b|$)",
        clean,
        re.I,
    )
    return (match.group(1) if match else clean.split(",", 1)[0]).strip()


def referent_from_sources(intent: str, sources: list[dict]) -> dict:
    """Build a compact, provenance-carrying referent from reviewed sources."""
    items: list[dict] = []
    source_ids: list[str] = []
    referent_type = None
    for source in sources:
        source_id = str(source.get("id") or "")
        categories = _category_facts([source])
        values: list[str] = []
        if intent == INTENT_PENDING:
            referent_type = "pending_item"
            values = [_pending_item(fact) for fact in categories[INTENT_PENDING]]
        elif intent == INTENT_MEDICATIONS:
            referent_type = "medication"
            values = _split_structured_values(_medication_values(categories))
        elif intent == INTENT_CONDITIONS:
            referent_type = "condition"
            values = _split_structured_values(_condition_values(categories))
        elif intent == INTENT_ALLERGIES:
            referent_type = "allergy"
            values = _split_structured_values(_allergy_values(categories))
        elif intent == INTENT_VITALS_LABS:
            referent_type = "result"
            values = _split_structured_values(_vital_values(categories))
        elif intent == INTENT_PLAN:
            referent_type = "plan"
            values = _split_structured_values(_plan_values(categories))
        elif intent == INTENT_LATEST_REPORT:
            referent_type = "report"
            values = [str(source.get("title") or "reviewed report")]
        for value in values:
            if not value:
                continue
            label = _medication_label(value) if intent == INTENT_MEDICATIONS else value
            item = {
                "value": value,
                "label": label,
                "source_id": source_id,
                "source_ids": [source_id] if source_id else [],
            }
            if intent == INTENT_LATEST_REPORT:
                item.update({
                    "title": str(source.get("title") or "reviewed report"),
                    "date": source.get("record_date"),
                    "document_type": source.get("document_type"),
                })
            label_tokens = _normalized_tokens(str(label))
            duplicate = None
            for existing in items:
                existing_tokens = _normalized_tokens(str(existing.get("label") or existing.get("value") or ""))
                if not label_tokens or not existing_tokens:
                    continue
                overlap = len(label_tokens & existing_tokens) / min(len(label_tokens), len(existing_tokens))
                if overlap >= 0.75:
                    duplicate = existing
                    break
            if duplicate is None:
                items.append(item)
            else:
                if len(str(label)) > len(str(duplicate.get("label") or "")):
                    duplicate["label"] = label
                    duplicate["value"] = value
                    duplicate["source_id"] = source_id or duplicate.get("source_id")
                merged_ids = list(dict.fromkeys((duplicate.get("source_ids") or []) + ([source_id] if source_id else [])))
                duplicate["source_ids"] = merged_ids
            if source_id and source_id not in source_ids:
                source_ids.append(source_id)
    if not referent_type or not items:
        if intent == INTENT_OVERVIEW:
            return {"type": "chart_overview", "items": [], "source_ids": [str(source.get("id")) for source in sources if source.get("id")]}
        return {}
    result = {"type": referent_type, "items": items[:8], "source_ids": source_ids}
    if len(items) == 1:
        result["selected"] = items[0]
    return result


def _selected_referent(dialogue: dict | None) -> dict:
    if not dialogue:
        return {}
    referent = dialogue.get("referent") or {}
    return referent.get("selected") or {}


def _medication_dose(value: str) -> str | None:
    match = re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b", value, re.I)
    return match.group(0) if match else None


def _medication_frequency(value: str) -> str | None:
    patterns = [
        r"\b(?:once|twice|three\s+times|four\s+times)\s+(?:a|per)\s+day\b",
        r"\bonce\s+daily\b", r"\btwice\s+daily\b", r"\bnightly\b",
        r"\beach\s+(?:morning|evening)\b", r"\bas\s+needed(?:\s+for\s+[^;,.]+)?",
        r"\bevery\s+\d+\s+hours?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, re.I)
        if match:
            return match.group(0)
    return None


def _structured_follow_up(question: str, intent: str, sources: list[dict], dialogue: dict | None) -> str | None:
    if not dialogue:
        return None
    relation = dialogue.get("relation")
    selected = _selected_referent(dialogue)
    value = str(selected.get("value") or "").strip()
    label = str(selected.get("label") or selected.get("name") or selected.get("title") or value).strip()

    if intent == INTENT_MEDICATIONS and selected:
        if relation == "dose":
            dose = _medication_dose(value)
            return f"The chart lists {label} at {dose}." if dose else f"The chart lists {value}, but I don’t see a separate dose documented."
        if relation == "frequency":
            frequency = _medication_frequency(value)
            return f"The chart says {label} is taken {frequency}." if frequency else f"The chart lists {value}, but I don’t see a separate frequency documented."
        if relation == "which":
            return f"You were asking about {label}."
        if relation in {"details", "clarification_selection"}:
            return f"The chart lists {value}."

    if intent == INTENT_PENDING and selected:
        display = _pending_display(value)
        if relation == "request_date":
            return f"The chart says that {display} is still pending, but it does not document when it was requested."
        if relation == "received_status":
            return f"The reviewed chart still lists {display} as pending. I don’t see a later reviewed entry saying it was received."
        if relation == "which":
            return f"The outstanding item is {display}."
        if relation in {"details", "clarification_selection"}:
            return f"The chart says that {display} is still pending."

    if intent == INTENT_LATEST_REPORT and relation == "changed_after" and selected:
        reference_date = str(selected.get("date") or "")
        latest = sources[0] if sources else None
        latest_date = str((latest or {}).get("record_date") or "")
        if not latest or not reference_date or latest_date <= reference_date:
            date_text = _format_date(reference_date)
            suffix = f" dated {date_text}" if date_text else ""
            return f"I don’t see a later reviewed report after the selected report{suffix}."
        summary = structured_chart_answer(question, INTENT_LATEST_REPORT, [latest], None)
        return f"After that report, {summary[0].lower() + summary[1:] if summary else 'a later reviewed report is available.'}"

    return None


def structured_chart_answer(question: str, intent: str, sources: list[dict], dialogue: dict | None = None) -> str | None:
    """Return a direct chart-native answer for common record questions."""
    if intent == INTENT_OPEN:
        return None
    follow_up = _structured_follow_up(question, intent, sources, dialogue)
    if follow_up is not None:
        return follow_up
    categories = _category_facts(sources)
    q = question.casefold()

    if intent == INTENT_PENDING:
        items = _unique_values(_pending_item(fact) for fact in categories[INTENT_PENDING])
        # Prefer specific named reports/tests over generic "outside report" wording.
        items.sort(key=lambda item: (0 if re.search(r"\b(?:cardiology|retinal|pulmonary|imaging|laboratory|specialist)\b", item, re.I) else 1, len(item)))
        if not items:
            return "I don’t see any reviewed record saying that a report, result, or test is currently pending."
        displayed = [_pending_display(item) for item in items[:3]]
        if re.search(r"\bwhen\b.*\brequest(?:ed)?\b", q):
            return f"The chart says that {displayed[0]} is still pending, but it does not document when it was requested."
        if re.search(r"\b(?:nothing\s+(?:is\s+)?missing|so\s+nothing|none\s+missing)\b", q):
            return f"No. The chart still lists {_format_join(displayed)} as pending."
        if len(displayed) == 1:
            return f"The chart says that {displayed[0]} is still pending."
        return f"The chart lists these outstanding items: {_format_join(displayed)}."

    if intent == INTENT_MEDICATIONS:
        values = _medication_values(categories)
        if not values:
            return "I don’t see a reviewed current-medication list in the available chart."
        return f"The current medications are {_format_join([_clean_list_text(value) for value in values[:3]])}."

    if intent == INTENT_ALLERGIES:
        values = _allergy_values(categories)
        if not values:
            return "I don’t see a reviewed medication-allergy entry in the available chart."
        return f"The chart documents {_format_join(values[:3])}."

    if intent == INTENT_CONDITIONS:
        conditions = _condition_values(categories)
        if not conditions:
            return "I don’t see a reviewed active-problem or diagnosis entry in the available chart."
        answer = f"The chart documents {_format_join(conditions[:3])}."
        status = _unique_values(categories["status"])
        if status:
            answer += " " + status[0].rstrip(".") + "."
        return answer

    if intent == INTENT_VITALS_LABS:
        values = _vital_values(categories)
        if not values:
            return "I don’t see reviewed vital-sign or laboratory values in the available chart."
        return f"The reviewed chart includes {_format_join(values[:4])}."

    if intent == INTENT_PLAN:
        values = _plan_values(categories)
        if not values:
            return "I don’t see a reviewed follow-up plan in the available chart."
        return f"The documented plan is to {_format_join(values[:3])}."

    if intent == INTENT_LATEST_REPORT:
        if not sources:
            return "I don’t see a reviewed report in the available chart."
        source = sources[0]
        date_text = _format_date(source.get("record_date"))
        type_text = source.get("document_type") or source.get("title") or "report"
        source_categories = _category_facts([source])
        pieces: list[str] = []
        conditions = _condition_values(source_categories)
        medications = _medication_values(source_categories)
        allergies = _allergy_values(source_categories)
        pending = _unique_values(_pending_item(fact) for fact in source_categories[INTENT_PENDING])
        if conditions:
            pieces.append(f"It documents {_format_join(conditions[:2])}.")
        if medications:
            pieces.append(f"It lists {_format_join(medications[:2])}.")
        if allergies:
            pieces.append(f"It notes {_format_join(allergies[:1])}.")
        if pending:
            pieces.append(f"It says that {_format_join([_pending_display(item) for item in pending[:2]])} is still pending.")
        status_values = _unique_values(source_categories["status"])
        for status_value in status_values[:2]:
            sentence = status_value.rstrip(".")
            if sentence and not any(sentence.casefold() in piece.casefold() for piece in pieces):
                pieces.append(sentence[0].upper() + sentence[1:] + ".")
        heading = f"The latest reviewed report is the {type_text}"
        if date_text:
            heading += f" dated {date_text}"
        heading += "."
        return " ".join([heading] + pieces[:4])

    if intent == INTENT_OVERVIEW:
        parts: list[str] = []
        conditions = _condition_values(categories)
        medications = _medication_values(categories)
        allergies = _allergy_values(categories)
        pending = _unique_values(_pending_item(fact) for fact in categories[INTENT_PENDING])
        if conditions:
            parts.append(f"The chart documents {_format_join(conditions[:2])}.")
        if medications:
            parts.append(f"Current medications are {_format_join(medications[:2])}.")
        if allergies:
            parts.append(f"It notes {_format_join(allergies[:1])}.")
        if pending:
            label = "item" if len(pending) == 1 else "items"
            verb = "is" if len(pending) == 1 else "are"
            parts.append(f"The remaining outstanding {label} {verb} {_format_join([_pending_display(item) for item in pending[:2]])}.")
        if not parts:
            return "I don’t have enough reviewed information in this chart to summarize the patient."
        return " ".join(parts[:4])

    return None


def requested_word_limit(question: str) -> int | None:
    match = re.search(r"\b(?:in|under|about|max(?:imum)?(?:\s+of)?)\s+(\d{1,3})\s+words?\b", question, re.I)
    if not match:
        return None
    return max(5, min(int(match.group(1)), 150))


def truncate_words(text: str, limit: int | None) -> str:
    if not limit:
        return text.strip()
    words = text.strip().split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).rstrip(" ,;:") + "."


def answer_looks_like_source_dump(answer: str) -> bool:
    lower = answer.casefold()
    if len(answer) > 1400:
        return True
    if any(marker in lower for marker in DEMO_MARKERS):
        return True
    if re.search(r"(?im)^\s*(?:patient|dob|mrn|report date|document type)\s*:", answer):
        return True
    sentences = [re.sub(r"\W+", " ", sentence.casefold()).strip() for sentence in re.split(r"(?<=[.!?])\s+", answer) if len(sentence.strip()) > 20]
    if len(sentences) >= 3 and len(set(sentences)) <= max(1, len(sentences) // 2):
        return True
    # Repetition of a long phrase is a common failure mode of very small models.
    words = re.findall(r"\w+", lower)
    if len(words) >= 60:
        grams = [" ".join(words[index:index+12]) for index in range(len(words)-11)]
        if len(grams) != len(set(grams)):
            return True
    return False
