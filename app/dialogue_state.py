"""Deterministic dialogue-state resolution for chart conversations.

The resolver interprets follow-up language and requested response format without
using the local language model. It never treats previous assistant text as chart
evidence. All factual answers are still assembled from freshly loaded reviewed
records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from .chart_assistant import (
    INTENT_ALLERGIES,
    INTENT_CONDITIONS,
    INTENT_LATEST_REPORT,
    INTENT_MEDICATIONS,
    INTENT_OPEN,
    INTENT_OVERVIEW,
    INTENT_PENDING,
    INTENT_PLAN,
    INTENT_VITALS_LABS,
    direct_intent,
    referent_from_sources,
    requested_word_limit,
)

INTENT_LABELS = {
    INTENT_PENDING: "pending reports or tests",
    INTENT_MEDICATIONS: "medications",
    INTENT_CONDITIONS: "conditions",
    INTENT_LATEST_REPORT: "latest report",
    INTENT_OVERVIEW: "chart overview",
    INTENT_ALLERGIES: "allergies",
    INTENT_VITALS_LABS: "vitals or laboratory values",
    INTENT_PLAN: "follow-up plan",
    INTENT_OPEN: "open chart question",
}

_RECORD_LOOKUP_RE = re.compile(
    r"\b(?:chart|record|records|documented|listed|prescribed|medication\s+list|"
    r"according\s+to|does\s+the\s+chart|what\s+does\s+the\s+record)\b",
    re.I,
)
_TREATMENT_PATTERNS = [
    re.compile(r"\bwhat\s+(?:can|could|should)\s+(?:i|the\s+patient|they|he|she|we)\s+(?:take|use|do)\b", re.I),
    re.compile(r"\bwhat\s+to\s+take\b", re.I),
    re.compile(r"\b(?:what|which)\s+(?:medicine|medication|drug|treatment)\b.*\b(?:for|should|recommend)\b", re.I),
    re.compile(r"\b(?:recommend|prescribe|best\s+medicine|treatment\s+options?|medication\s+options?)\b", re.I),
    re.compile(r"\bhow\s+much\b.*\b(?:take|dose|dosage)\b", re.I),
]

RELATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("request_date", re.compile(r"\bwhen\b.*\b(?:request(?:ed)?|order(?:ed)?)\b", re.I)),
    ("received_status", re.compile(r"\b(?:was|has|did)\b.*\b(?:received|returned|completed|come\s+back|available)\b|\b(?:ever\s+)?received\b", re.I)),
    ("dose", re.compile(r"\b(?:what\s+)?dose\b|\bdosage\b|\bhow\s+much\b|\bstrength\b", re.I)),
    ("frequency", re.compile(r"\bhow\s+often\b|\bfrequency\b|\bwhen\s+(?:is|was|should)\b.*\b(?:take|taken|use|used)\b|\btimes?\s+(?:a|per)\s+day\b", re.I)),
    ("changed_after", re.compile(r"\bwhat\s+changed\s+after\b|\bwhat\s+happened\s+(?:after|next)\b|\banything\s+new\s+after\b", re.I)),
    ("which", re.compile(r"^\s*(?:which\s+one|which\s+item|the\s+first\s+one|the\s+second\s+one|first\s+one|second\s+one|former|latter)\??\s*$", re.I)),
    ("confirmation", re.compile(r"^\s*(?:so\s+)?(?:nothing|none)\s+(?:is\s+)?(?:missing|pending)\??\s*$", re.I)),
    ("details", re.compile(r"^\s*(?:tell\s+me\s+more|more\s+details?|what\s+else|what\s+other\s+options(?:\s+are\s+there)?|other\s+options|alternatives?|anything\s+else|expand\s+on\s+that|what\s+about\s+(?:it|that|them))\??\s*$", re.I)),
    ("format_change", re.compile(r"^\s*(?:make\s+it\s+shorter|shorter|briefly|be\s+brief|in\s+bullet\s+points?|as\s+a\s+list|more\s+detail|expand\s+that)\.?\s*$", re.I)),
]

VAGUE_RE = re.compile(
    r"^\s*(?:so\??|really\??|are\s+you\s+sure\??|what\s+about\s+(?:it|that|them)\??|"
    r"tell\s+me\s+more\??|can\s+you\s+clarify\??|what\s+else\??|what\s+other\s+options(?:\s+are\s+there)?\??|other\s+options\??|alternatives?\??|which\s+one\??|"
    r"what\s+dose\??|how\s+often\??|when\s+was\s+it\s+requested\??|was\s+it\s+received\??)\s*$",
    re.I,
)

SINGLE_ITEM_RELATIONS = {
    "dose", "frequency", "request_date", "received_status", "which", "changed_after",
    "prescribed_date", "dispensed_date", "pickup_date", "started_date", "medication_event_date",
}


@dataclass
class DialogueResolution:
    intent: str = INTENT_OPEN
    relation: str = "new_question"
    referent: dict[str, Any] = field(default_factory=dict)
    requested_format: dict[str, Any] = field(default_factory=dict)
    safety_mode: str = "chart_only"
    confidence: float = 0.55
    clarification_required: bool = False
    clarification_question: str | None = None
    inherited: bool = False
    reason: str = "open_question"
    search_terms: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return asdict(self)

    def storage_format(self) -> dict[str, Any]:
        return {
            **self.requested_format,
            "resolver": {
                "confidence": round(float(self.confidence), 4),
                "relation": self.relation,
                "clarification_required": self.clarification_required,
                "clarification_question": self.clarification_question,
                "inherited": self.inherited,
                "reason": self.reason,
            },
        }


def default_state(chart_revision: int | None = None) -> dict[str, Any]:
    return {
        "active_intent": None,
        "referent": {},
        "requested_format": {},
        "safety_mode": "chart_only",
        "last_source_ids": [],
        "chart_revision": chart_revision,
        "resolver_confidence": 0.0,
        "relation": None,
        "clarification": {},
        "last_resolution": {},
    }


def _normalize_state(value: dict[str, Any] | None) -> dict[str, Any]:
    state = default_state()
    if value:
        state.update(value)
    state["referent"] = dict(state.get("referent") or {})
    state["requested_format"] = dict(state.get("requested_format") or {})
    state["clarification"] = dict(state.get("clarification") or {})
    return state


def _detect_relation(question: str) -> str:
    for relation, pattern in RELATION_PATTERNS:
        if pattern.search(question):
            return relation
    return "new_question"


def _record_lookup_requested(question: str) -> bool:
    return bool(_RECORD_LOOKUP_RE.search(question))


def _direct_treatment_request(question: str) -> bool:
    if _record_lookup_requested(question):
        return False
    return any(pattern.search(question) for pattern in _TREATMENT_PATTERNS)


def _parse_format(question: str, previous: dict[str, Any], *, inherited: bool, relation: str) -> dict[str, Any]:
    q = " ".join(question.casefold().split())
    explicit_limit = requested_word_limit(question)
    if explicit_limit:
        return {"word_limit": explicit_limit, "style": "brief" if explicit_limit <= 45 else "standard"}
    if re.search(r"\b(?:bullet\s+points?|as\s+a\s+list|list\s+format)\b", q):
        return {"style": "bullets"}
    if re.search(r"\b(?:briefly|concise|short\s+answer|be\s+brief)\b", q):
        return {"style": "brief", "word_limit": 40}
    if re.search(r"\b(?:make\s+it\s+shorter|shorter)\b", q):
        prior_limit = previous.get("word_limit")
        return {"style": "brief", "word_limit": max(10, min(int(prior_limit or 35), 35))}
    if re.search(r"\b(?:more\s+detail|expand\s+that|detailed)\b", q):
        return {"style": "detailed"}
    if inherited and relation not in {"new_question"}:
        return {key: value for key, value in previous.items() if key != "resolver"}
    return {}


def _items(referent: dict[str, Any]) -> list[dict[str, Any]]:
    values = referent.get("items") or []
    return [item for item in values if isinstance(item, dict)]


def _item_label(item: dict[str, Any]) -> str:
    return str(item.get("label") or item.get("name") or item.get("title") or item.get("value") or "item").strip()


def _select_item(question: str, referent: dict[str, Any], clarification: dict[str, Any] | None = None) -> dict[str, Any] | None:
    candidates = _items(referent)
    if not candidates and clarification:
        candidates = [item for item in clarification.get("options", []) if isinstance(item, dict)]
    if not candidates:
        return None
    q = " ".join(question.casefold().split())
    if re.fullmatch(r"(?:both|all|all of them|both of them)[.?!]*", q):
        return {
            "label": "both" if len(candidates) == 2 else "all",
            "value": "both" if len(candidates) == 2 else "all",
            "selection": "all",
            "items": candidates,
        }
    ordinal = None
    if re.search(r"\b(?:first|former|1st)\b", q):
        ordinal = 0
    elif re.search(r"\b(?:second|latter|2nd)\b", q):
        ordinal = 1
    elif re.search(r"\b(?:third|3rd)\b", q):
        ordinal = 2
    if ordinal is not None and ordinal < len(candidates):
        return candidates[ordinal]
    scored: list[tuple[int, dict[str, Any]]] = []
    q_tokens = set(re.findall(r"[a-z0-9][a-z0-9-]{2,}", q))
    for item in candidates:
        label = _item_label(item).casefold()
        value = str(item.get("value") or "").casefold()
        if label and label in q:
            return item
        tokens = set(re.findall(r"[a-z0-9][a-z0-9-]{2,}", label + " " + value))
        scored.append((len(q_tokens & tokens), item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def _clarification_for_options(intent: str, relation: str, referent: dict[str, Any]) -> dict[str, Any]:
    options = _items(referent)[:4]
    labels = [_item_label(item) for item in options]
    if not labels:
        question = "Which chart item do you mean?"
    elif len(labels) == 1:
        question = f"Do you mean {labels[0]}?"
    elif len(labels) == 2:
        question = f"Which one do you mean—{labels[0]} or {labels[1]}?"
    else:
        question = "Which one do you mean: " + ", ".join(labels[:-1]) + f", or {labels[-1]}?"
    return {
        "required": True,
        "question": question,
        "intent": intent,
        "pending_relation": relation,
        "options": options,
    }


def _generic_clarification() -> dict[str, Any]:
    return {
        "required": True,
        "question": (
            "What would you like me to check in the chart—medications, conditions, pending reports or tests, "
            "allergies, vitals or labs, the latest report, or the follow-up plan?"
        ),
        "intent": None,
        "pending_relation": "new_question",
        "options": [],
    }


def resolve_dialogue_state(
    question: str,
    prior_state: dict[str, Any] | None,
    history: list[dict[str, Any]] | None = None,
) -> DialogueResolution:
    state = _normalize_state(prior_state)
    history = history or []
    relation = _detect_relation(question)
    direct = direct_intent(question)
    prior_intent = state.get("active_intent")
    prior_referent = state.get("referent") or {}
    prior_clarification = state.get("clarification") or {}
    if not prior_intent:
        for turn in reversed(history):
            if str(turn.get("role") or "") != "user":
                continue
            inherited_intent = direct_intent(str(turn.get("content") or ""))
            if inherited_intent:
                prior_intent = inherited_intent
                break
    legacy_treatment_context = any(
        str(turn.get("role") or "") == "user" and _direct_treatment_request(str(turn.get("content") or ""))
        for turn in history[-6:]
    )

    # A reply to a previous clarification is resolved before ordinary routing.
    if prior_clarification.get("required"):
        selected = _select_item(question, prior_referent, prior_clarification)
        if selected:
            referent = dict(prior_referent)
            referent["selected"] = selected
            intent = str(prior_clarification.get("intent") or prior_intent or INTENT_OPEN)
            pending_relation = str(prior_clarification.get("pending_relation") or "details")
            return DialogueResolution(
                intent=intent,
                relation=pending_relation,
                referent=referent,
                requested_format=_parse_format(question, state.get("requested_format") or {}, inherited=True, relation=pending_relation),
                safety_mode=str(state.get("safety_mode") or "chart_only"),
                confidence=0.98,
                clarification_required=False,
                inherited=True,
                reason="clarification_selection",
                search_terms=[_item_label(selected), str(selected.get("value") or "")],
            )
        if relation == "new_question" and direct:
            # A new explicit question replaces the old clarification request.
            prior_clarification = {}
        else:
            question_text = str(prior_clarification.get("question") or "Which item do you mean?")
            return DialogueResolution(
                intent=str(prior_clarification.get("intent") or prior_intent or INTENT_OPEN),
                relation=str(prior_clarification.get("pending_relation") or relation),
                referent=prior_referent,
                requested_format=_parse_format(question, state.get("requested_format") or {}, inherited=True, relation=relation),
                safety_mode=str(state.get("safety_mode") or "chart_only"),
                confidence=0.35,
                clarification_required=True,
                clarification_question=question_text,
                inherited=True,
                reason="clarification_unresolved",
            )

    if _direct_treatment_request(question):
        return DialogueResolution(
            intent=direct or prior_intent or INTENT_OPEN,
            relation=relation,
            referent=prior_referent if direct is None else {},
            requested_format=_parse_format(question, state.get("requested_format") or {}, inherited=direct is None, relation=relation),
            safety_mode="treatment_boundary",
            confidence=0.99,
            inherited=direct is None,
            reason="direct_treatment_request",
        )

    inherited = False
    if direct:
        intent = direct
        confidence = 0.98
        reason = "explicit_intent"
        referent = prior_referent if direct == prior_intent else {}
        safety_mode = "chart_only"
    elif prior_intent and (relation != "new_question" or VAGUE_RE.match(question)):
        intent = str(prior_intent)
        inherited = True
        confidence = 0.92 if relation != "new_question" else 0.78
        reason = "inherited_dialogue_state" if state.get("active_intent") else "inherited_legacy_history"
        if not state.get("active_intent"):
            confidence = min(confidence, 0.84)
        referent = prior_referent
        safety_mode = str(state.get("safety_mode") or "chart_only")
    elif (state.get("safety_mode") == "treatment_boundary" or legacy_treatment_context) and VAGUE_RE.match(question):
        intent = str(prior_intent or INTENT_OPEN)
        inherited = True
        confidence = 0.9
        reason = "inherited_treatment_boundary"
        referent = prior_referent
        safety_mode = "treatment_boundary"
    elif VAGUE_RE.match(question):
        clarification = _generic_clarification()
        return DialogueResolution(
            intent=INTENT_OPEN,
            relation=relation,
            referent={},
            requested_format=_parse_format(question, {}, inherited=False, relation=relation),
            safety_mode="chart_only",
            confidence=0.2,
            clarification_required=True,
            clarification_question=clarification["question"],
            inherited=False,
            reason="no_resolvable_context",
        )
    else:
        intent = INTENT_OPEN
        confidence = 0.55
        reason = "open_question"
        referent = {}
        safety_mode = "chart_only"

    selected = _select_item(question, referent)
    if selected:
        referent = dict(referent)
        referent["selected"] = selected
        confidence = max(confidence, 0.96)
        reason = "explicit_referent_selection"

    requested_format = _parse_format(
        question,
        state.get("requested_format") or {},
        inherited=inherited or relation == "format_change",
        relation=relation,
    )
    search_terms: list[str] = []
    if referent.get("selected"):
        search_terms.extend([_item_label(referent["selected"]), str(referent["selected"].get("value") or "")])

    return DialogueResolution(
        intent=intent,
        relation=relation,
        referent=referent,
        requested_format=requested_format,
        safety_mode=safety_mode,
        confidence=confidence,
        inherited=inherited,
        reason=reason,
        search_terms=[term for term in search_terms if term],
    )


def enrich_dialogue_resolution(
    resolution: DialogueResolution,
    sources: list[dict[str, Any]],
    question: str,
) -> DialogueResolution:
    # Medication-event clarification options (for example dispensed vs picked
    # up) are workflow choices, not chart entities. Preserve them instead of
    # replacing them with the current medication-list referent before the
    # medication-event workflow consumes the selection.
    if (
        resolution.relation == "medication_event_date"
        and (resolution.referent or {}).get("type") == "medication_event_choice"
    ):
        return resolution
    if resolution.clarification_required and resolution.referent:
        return resolution
    evidence_referent = referent_from_sources(resolution.intent, sources)
    referent = dict(resolution.referent or {})
    if evidence_referent:
        if not referent or referent.get("type") != evidence_referent.get("type"):
            referent = evidence_referent
        else:
            existing = {_item_label(item).casefold(): item for item in _items(referent)}
            for item in _items(evidence_referent):
                existing.setdefault(_item_label(item).casefold(), item)
            referent["items"] = list(existing.values())
            for key in ["type", "source_ids"]:
                if evidence_referent.get(key):
                    referent[key] = evidence_referent[key]

    selected = _select_item(question, referent)
    if selected:
        referent["selected"] = selected
    elif referent.get("selected"):
        selected = referent.get("selected")
    elif len(_items(referent)) == 1:
        selected = _items(referent)[0]
        referent["selected"] = selected

    if resolution.relation in SINGLE_ITEM_RELATIONS and len(_items(referent)) > 1 and not selected:
        clarification = _clarification_for_options(resolution.intent, resolution.relation, referent)
        resolution.clarification_required = True
        resolution.clarification_question = clarification["question"]
        resolution.confidence = min(resolution.confidence, 0.55)
        resolution.reason = "multiple_possible_referents"

    resolution.referent = referent
    resolution.search_terms = []
    if selected:
        resolution.search_terms = [_item_label(selected), str(selected.get("value") or "")]
    return resolution


def resolution_clarification(resolution: DialogueResolution) -> dict[str, Any]:
    if not resolution.clarification_required:
        return {}
    options = _items(resolution.referent)[:4]
    return {
        "required": True,
        "question": resolution.clarification_question,
        "intent": resolution.intent,
        "pending_relation": resolution.relation,
        "options": options,
    }


def apply_requested_format(answer: str, requested_format: dict[str, Any]) -> str:
    text = " ".join(answer.strip().split())
    style = requested_format.get("style")
    word_limit = requested_format.get("word_limit")
    if style == "bullets":
        body = text
        prefix = None
        match = re.match(r"^(The current medications are|The chart documents|The reviewed chart includes|The documented plan is to)\s+(.+?)\.?$", body, re.I)
        if match:
            prefix = match.group(1).rstrip()
            body = match.group(2)
        values = [value.strip(" .") for value in re.split(r"\s*;\s*|,\s+and\s+|\s+and\s+(?=[A-Z0-9])", body) if value.strip(" .")]
        if len(values) > 1:
            heading = (prefix + ":\n") if prefix else ""
            text = heading + "\n".join(f"- {value}" for value in values)
    if style == "brief" and not word_limit:
        word_limit = 40
    if word_limit:
        words = text.split()
        limit = max(5, min(int(word_limit), 150))
        if len(words) > limit:
            text = " ".join(words[:limit]).rstrip(" ,;:") + "."
    return text
