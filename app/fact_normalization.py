"""Conservative normalization helpers for provenance-backed chart facts.

Normalization enriches derived facts for search/evaluation. It never changes the
source quotation and never promotes a weaker statement (for example, merely
"documented in records") into a stronger current/active clinical assertion.
"""
from __future__ import annotations

import re
from typing import Any

_DOSE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)\b", re.I)
_FORM_WORDS = (
    "tablet", "tablets", "capsule", "capsules", "inhaler", "spray", "solution",
    "suspension", "cream", "ointment", "patch", "injection", "syrup",
)
_SCHEDULE_PATTERNS = [
    r"\b(?:once|twice|three times)\s+daily\b",
    r"\bdaily\b",
    r"\bnightly\b",
    r"\beach morning\b",
    r"\bas needed(?:\s+for\s+[^,;.]+)?\b",
    r"\bwith meals\b",
    r"\bduring allergy season\b",
]


def _clean(value: str) -> str:
    return " ".join(str(value or "").split()).strip(" .;:")


def normalize_code(code: str | None) -> dict[str, Any]:
    value = _clean(code or "")
    if not value:
        return {}
    system = None
    if re.fullmatch(r"[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?", value, re.I):
        system = "ICD-10-CM"
    elif re.fullmatch(r"\d{1,7}-\d", value):
        system = "LOINC"
    return {"code": value, "code_system": system}


def normalize_condition(value: str, *, code: str | None = None, status: str = "active") -> dict[str, Any]:
    display = _clean(value)
    result: dict[str, Any] = {"display": display, "status": status}
    result.update(normalize_code(code))
    return result


def normalize_medication(value: str, *, status: str) -> dict[str, Any]:
    text = _clean(value)
    dose_match = _DOSE_RE.search(text)
    dose = None
    if dose_match:
        dose = {"value": float(dose_match.group(1)), "unit": dose_match.group(2).casefold()}
    before_dose = text[:dose_match.start()].strip() if dose_match else text
    # Remove common leading clinical qualifiers without guessing RxNorm identity.
    before_dose = re.sub(r"(?i)^(?:current|documented|historical|past|previous)\s+", "", before_dose).strip()
    name = before_dose
    form = None
    lower = text.casefold()
    for candidate in _FORM_WORDS:
        if re.search(rf"\b{re.escape(candidate)}\b", lower):
            form = candidate.rstrip("s")
            if name.casefold().endswith(" " + candidate):
                name = name[: -(len(candidate) + 1)].strip()
            break
    schedule = None
    for pattern in _SCHEDULE_PATTERNS:
        match = re.search(pattern, text, re.I)
        if match:
            schedule = match.group(0)
            break
    return {
        "name": name or text,
        "dose": dose,
        "form": form,
        "schedule": schedule,
        "as_needed": bool(re.search(r"\bas needed\b|\bPRN\b", text, re.I)),
        "status": status,
        "display": text,
    }


def normalize_measurement(name: str, value: str, *, code: str | None = None, status: str | None = None) -> dict[str, Any]:
    display_name = _clean(name)
    raw = _clean(value)
    number_match = re.match(r"^(-?\d+(?:\.\d+)?)\s*(.*)$", raw)
    result: dict[str, Any] = {
        "name": display_name,
        "value": raw,
        "result_status": _clean(status or "") or None,
    }
    if number_match:
        result["numeric_value"] = float(number_match.group(1))
        result["unit"] = _clean(number_match.group(2)) or None
    result.update(normalize_code(code))
    return result


def normalize_allergy(value: str) -> dict[str, Any]:
    text = _clean(value)
    if re.search(r"(?i)\b(?:no known drug allergies|no medication allergies|none documented)\b", text):
        return {"status": "none_documented", "display": text, "substance": None, "reaction": None}
    match = re.match(r"(?i)^(.+?)(?:\s+allergy)?\s+(?:with|caused|causes?)\s+(.+)$", text)
    if match:
        return {"status": "documented", "display": text, "substance": _clean(match.group(1)), "reaction": _clean(match.group(2))}
    return {"status": "documented", "display": text, "substance": text, "reaction": None}


def normalize_pending(value: str) -> dict[str, Any]:
    text = _clean(value)
    lower = text.casefold()
    item_type = "item"
    for candidate in ("report", "test", "testing", "result", "referral", "consultation", "consult", "study", "imaging", "records"):
        if re.search(rf"\b{candidate}\b", lower):
            item_type = candidate
            break
    return {"display": text, "item_type": item_type, "status": "pending"}


def normalize_plan(value: str) -> dict[str, Any]:
    text = _clean(value)
    actions = []
    for verb in ("continue", "return", "repeat", "obtain", "monitor", "keep", "follow up", "follow-up", "schedule", "review"):
        if re.search(rf"\b{re.escape(verb)}\b", text, re.I):
            actions.append(verb.replace("-", " "))
    return {"display": text, "actions": sorted(set(actions))}
