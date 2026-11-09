from __future__ import annotations
from datetime import date, datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PasswordBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(min_length=1, max_length=256)


class Setup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vault_password: str = Field(min_length=14, max_length=256)
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=12, max_length=256)


class Login(PasswordBody):
    username: str = Field(min_length=3, max_length=50)


class UserCreate(Login):
    username: str = Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=12, max_length=256)
    role: Literal["doctor", "staff"]


class UserActive(Strict):
    active: bool
    reason: str = Field(min_length=5, max_length=500)


class PatientCreate(Strict):
    name: str = Field(min_length=2, max_length=120)
    dob: date
    mrn: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")

    @field_validator("dob")
    @classmethod
    def past_date(cls, value):
        if value > date.today() or value.year < 1850:
            raise ValueError("Date of birth must be a plausible past date.")
        return value


class PatientUpdate(PatientCreate):
    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=5, max_length=500)


class Assignment(Strict):
    user_id: str = Field(min_length=1, max_length=64)
    enabled: bool = True
    reason: str = Field(min_length=5, max_length=500)


class SessionCreate(Strict):
    title: str = Field(min_length=2, max_length=160)
    session_type: Literal["office", "telehealth", "phone", "records_review", "procedure", "other"]
    status: Literal["planned", "open", "closed", "cancelled"] = "planned"
    starts_at: datetime
    ends_at: datetime | None = None
    location: str = Field(default="", max_length=240)
    clinician_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def valid_times(self):
        if self.starts_at.tzinfo is None or self.starts_at.utcoffset() is None:
            raise ValueError("starts_at must include a time-zone offset.")
        if self.ends_at is not None:
            if self.ends_at.tzinfo is None or self.ends_at.utcoffset() is None:
                raise ValueError("ends_at must include a time-zone offset.")
            if self.ends_at <= self.starts_at:
                raise ValueError("ends_at must be after starts_at.")
        return self


class SessionUpdate(SessionCreate):
    expected_version: int = Field(ge=1)
    reason: str = Field(min_length=5, max_length=500)


class ReviewDocument(Strict):
    patient_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=160000)
    summary: str = Field(min_length=1, max_length=12000)
    expected_version: int = Field(ge=1)
    reason: str = Field(min_length=5, max_length=500)
    identity_confirmed: Literal[True]
    content_confirmed: Literal[True]


class ContextChange(Strict):
    enabled: bool
    expected_version: int = Field(ge=1)
    reason: str = Field(min_length=5, max_length=500)


class NoteCreate(Strict):
    title: str = Field(min_length=2, max_length=160)
    body: str = Field(min_length=1, max_length=30000)


class NoteUpdate(NoteCreate):
    expected_version: int = Field(ge=1)
    reason: str = Field(min_length=5, max_length=500)


class AssistantThreadOpen(Strict):
    scope: Literal["longitudinal", "session"] = "longitudinal"
    session_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def valid_scope(self):
        if self.scope == "longitudinal" and self.session_id is not None:
            raise ValueError("Longitudinal assistant thread cannot have a session_id.")
        if self.scope == "session" and not self.session_id:
            raise ValueError("Session assistant thread requires session_id.")
        return self


class AssistantThreadClear(Strict):
    reason: str = Field(min_length=5, max_length=500)


class ConversationTurn(Strict):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class Question(Strict):
    question: str = Field(min_length=2, max_length=1500)
    expected_revision: int = Field(ge=1)
    thread_id: str | None = Field(default=None, max_length=64)
    history: list[ConversationTurn] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def bounded_history(self):
        if sum(len(turn.content) for turn in self.history) > 8000:
            raise ValueError("Conversation history is too large.")
        return self


class ToolInvocation(Strict):
    name: Literal["get_patient_context", "list_patient_documents"]
    arguments: dict
