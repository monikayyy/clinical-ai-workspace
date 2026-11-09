"""Encrypted, single-process SQLite snapshot store.

SQLite exists only in process memory. The complete serialized database, including
original files and audit events, is persisted with AES-256-GCM. Schema version 4 adds an encrypted provenance fact index on top of encrypted assistant threads and
the Stage 2 session foundation. Stage 1 through schema-version-3 vaults migrate after
successful decryption and audit verification without changing reviewed chart evidence.

This remains a synthetic-data engineering prototype, not a production EHR or a
substitute for security, privacy, clinical, or compliance review.
"""
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"CWSVAULT1\x00"
MAX_DB_BYTES = 96 * 1024 * 1024
SESSION_FOUNDATION_VERSION = 2
ASSISTANT_MEMORY_VERSION = 3
PROVENANCE_FACT_VERSION = 4
SCHEMA_VERSION = 5
SESSION_FOUNDATION_DESCRIPTION = "Stage 2 session foundation"
ASSISTANT_MEMORY_DESCRIPTION = "Stage 2 encrypted assistant conversation memory"
PROVENANCE_FACT_DESCRIPTION = "Stage 2 provenance fact store"
FACT_SEMANTICS_DESCRIPTION = "Stage 2 provenance conflict and lifecycle semantics"


class UnsupportedSchemaError(ValueError):
    """The decrypted vault is valid but does not match a supported schema."""


SCHEMA = f"""
PRAGMA foreign_keys=ON;
PRAGMA temp_store=MEMORY;
CREATE TABLE users (
 id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE, role TEXT NOT NULL,
 password_hash TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL
);
CREATE TABLE patients (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, dob TEXT NOT NULL, mrn TEXT NOT NULL UNIQUE,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE grants (
 patient_id TEXT NOT NULL REFERENCES patients(id), user_id TEXT NOT NULL REFERENCES users(id),
 PRIMARY KEY(patient_id,user_id)
);
CREATE TABLE sessions (
 id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id),
 title TEXT NOT NULL,
 session_type TEXT NOT NULL CHECK(session_type IN ('office','telehealth','phone','records_review','procedure','other')),
 status TEXT NOT NULL CHECK(status IN ('planned','open','closed','cancelled')),
 starts_at TEXT NOT NULL, ends_at TEXT, location TEXT NOT NULL DEFAULT '',
 clinician_id TEXT REFERENCES users(id), created_by TEXT NOT NULL REFERENCES users(id),
 version INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX sessions_patient_idx ON sessions(patient_id, starts_at DESC);
CREATE TABLE documents (
 id TEXT PRIMARY KEY, patient_id TEXT REFERENCES patients(id), session_id TEXT REFERENCES sessions(id),
 filename TEXT NOT NULL, mime TEXT NOT NULL, original BLOB NOT NULL, sha256 TEXT NOT NULL, text TEXT NOT NULL,
 summary TEXT NOT NULL, summary_kind TEXT NOT NULL, method TEXT NOT NULL,
 identity_json TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'review',
 context_enabled INTEGER NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 1,
 created_by TEXT NOT NULL REFERENCES users(id), created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX documents_patient_idx ON documents(patient_id);
CREATE INDEX documents_session_idx ON documents(session_id);
CREATE TABLE notes (
 id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id), session_id TEXT REFERENCES sessions(id),
 title TEXT NOT NULL, body TEXT NOT NULL, created_by TEXT NOT NULL REFERENCES users(id),
 version INTEGER NOT NULL DEFAULT 1, context_enabled INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX notes_session_idx ON notes(session_id);
CREATE TABLE note_versions (
 note_id TEXT NOT NULL REFERENCES notes(id), version INTEGER NOT NULL, title TEXT NOT NULL,
 body TEXT NOT NULL, editor TEXT NOT NULL REFERENCES users(id), reason TEXT NOT NULL,
 at TEXT NOT NULL, PRIMARY KEY(note_id,version)
);
CREATE TABLE schema_migrations (
 version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL, description TEXT NOT NULL
);
CREATE TABLE assistant_threads (
 id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
 patient_id TEXT NOT NULL REFERENCES patients(id), session_id TEXT REFERENCES sessions(id),
 scope TEXT NOT NULL CHECK(scope IN ('longitudinal','session')),
 title TEXT NOT NULL DEFAULT 'Chart conversation',
 status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','archived')),
 chart_revision INTEGER NOT NULL, version INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
 CHECK((scope='longitudinal' AND session_id IS NULL) OR (scope='session' AND session_id IS NOT NULL))
);
CREATE UNIQUE INDEX assistant_threads_longitudinal_unique
 ON assistant_threads(user_id,patient_id) WHERE scope='longitudinal' AND status='active';
CREATE UNIQUE INDEX assistant_threads_session_unique
 ON assistant_threads(user_id,patient_id,session_id) WHERE scope='session' AND status='active';
CREATE INDEX assistant_threads_user_patient_idx
 ON assistant_threads(user_id,patient_id,updated_at DESC);
CREATE TABLE assistant_messages (
 id TEXT PRIMARY KEY, thread_id TEXT NOT NULL REFERENCES assistant_threads(id) ON DELETE CASCADE,
 sequence INTEGER NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant')),
 content TEXT NOT NULL, answer_origin TEXT, source_ids_json TEXT NOT NULL DEFAULT '[]',
 chart_revision INTEGER NOT NULL, created_at TEXT NOT NULL,
 UNIQUE(thread_id,sequence)
);
CREATE INDEX assistant_messages_thread_idx ON assistant_messages(thread_id,sequence);
CREATE TABLE assistant_dialogue_state (
 thread_id TEXT PRIMARY KEY REFERENCES assistant_threads(id) ON DELETE CASCADE,
 active_intent TEXT, referent_json TEXT NOT NULL DEFAULT '{{}}',
 requested_format_json TEXT NOT NULL DEFAULT '{{}}',
 safety_mode TEXT NOT NULL DEFAULT 'chart_only',
 last_source_ids_json TEXT NOT NULL DEFAULT '[]',
 chart_revision INTEGER NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE clinical_fact_sources (
 source_kind TEXT NOT NULL CHECK(source_kind IN ('document','note')),
 source_id TEXT NOT NULL,
 patient_id TEXT NOT NULL REFERENCES patients(id),
 source_version INTEGER NOT NULL,
 source_sha256 TEXT NOT NULL,
 extraction_version TEXT NOT NULL,
 fact_count INTEGER NOT NULL,
 indexed_at TEXT NOT NULL,
 PRIMARY KEY(source_kind,source_id)
);
CREATE INDEX clinical_fact_sources_patient_idx ON clinical_fact_sources(patient_id);
CREATE TABLE clinical_facts (
 id TEXT PRIMARY KEY,
 patient_id TEXT NOT NULL REFERENCES patients(id),
 session_id TEXT REFERENCES sessions(id),
 fact_type TEXT NOT NULL,
 entity_key TEXT,
 entity_label TEXT,
 event_type TEXT,
 event_role TEXT NOT NULL DEFAULT 'unspecified',
 value_text TEXT NOT NULL,
 normalized_json TEXT NOT NULL DEFAULT '{{}}',
 event_date TEXT,
 asserted_at TEXT,
 effective_start TEXT,
 effective_end TEXT,
 assertion_key TEXT NOT NULL DEFAULT '',
 supersedes_json TEXT NOT NULL DEFAULT '[]',
 status TEXT NOT NULL DEFAULT 'current',
 polarity TEXT NOT NULL DEFAULT 'affirmed',
 certainty TEXT NOT NULL DEFAULT 'explicit',
 source_kind TEXT NOT NULL CHECK(source_kind IN ('document','note')),
 source_id TEXT NOT NULL,
 source_version INTEGER NOT NULL,
 source_sha256 TEXT NOT NULL,
 source_title TEXT NOT NULL,
 quote TEXT NOT NULL,
 start_offset INTEGER NOT NULL,
 end_offset INTEGER NOT NULL,
 extraction_method TEXT NOT NULL,
 extraction_version TEXT NOT NULL,
 semantic_version TEXT NOT NULL DEFAULT 'fact-semantics-v1',
 created_at TEXT NOT NULL,
 UNIQUE(source_kind,source_id,source_version,fact_type,entity_key,event_type,event_role,start_offset,end_offset,value_text)
);
CREATE INDEX clinical_facts_patient_type_idx ON clinical_facts(patient_id,fact_type);
CREATE INDEX clinical_facts_patient_entity_idx ON clinical_facts(patient_id,entity_key);
CREATE INDEX clinical_facts_patient_event_idx ON clinical_facts(patient_id,event_type,event_role,event_date);
CREATE INDEX clinical_facts_source_idx ON clinical_facts(source_kind,source_id,source_version);
CREATE INDEX clinical_facts_assertion_idx ON clinical_facts(patient_id,assertion_key);
CREATE TABLE clinical_fact_resolutions (
 id TEXT PRIMARY KEY,
 patient_id TEXT NOT NULL REFERENCES patients(id),
 resolution_key TEXT NOT NULL,
 fact_type TEXT NOT NULL,
 entity_key TEXT,
 event_type TEXT,
 event_role TEXT NOT NULL DEFAULT 'unspecified',
 status TEXT NOT NULL,
 preferred_fact_id TEXT,
 fact_ids_json TEXT NOT NULL DEFAULT '[]',
 superseded_fact_ids_json TEXT NOT NULL DEFAULT '[]',
 values_json TEXT NOT NULL DEFAULT '{{}}',
 computed_at TEXT NOT NULL,
 UNIQUE(patient_id,resolution_key)
);
CREATE INDEX clinical_fact_resolutions_patient_idx ON clinical_fact_resolutions(patient_id,fact_type,status);
CREATE TABLE audit (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, actor TEXT NOT NULL,
 patient_id TEXT, action TEXT NOT NULL, entity_id TEXT NOT NULL, details TEXT NOT NULL,
 previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL
);
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit BEGIN SELECT RAISE(ABORT,'append-only audit'); END;
CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit BEGIN SELECT RAISE(ABORT,'append-only audit'); END;
INSERT INTO schema_migrations(version,applied_at,description)
VALUES(1,strftime('%Y-%m-%dT%H:%M:%fZ','now'),'Stage 1 records foundation');
INSERT INTO schema_migrations(version,applied_at,description)
VALUES({SESSION_FOUNDATION_VERSION},strftime('%Y-%m-%dT%H:%M:%fZ','now'),'{SESSION_FOUNDATION_DESCRIPTION}');
INSERT INTO schema_migrations(version,applied_at,description)
VALUES({ASSISTANT_MEMORY_VERSION},strftime('%Y-%m-%dT%H:%M:%fZ','now'),'{ASSISTANT_MEMORY_DESCRIPTION}');
INSERT INTO schema_migrations(version,applied_at,description)
VALUES({PROVENANCE_FACT_VERSION},strftime('%Y-%m-%dT%H:%M:%fZ','now'),'{PROVENANCE_FACT_DESCRIPTION}');
INSERT INTO schema_migrations(version,applied_at,description)
VALUES({SCHEMA_VERSION},strftime('%Y-%m-%dT%H:%M:%fZ','now'),'{FACT_SEMANTICS_DESCRIPTION}');
PRAGMA user_version={SCHEMA_VERSION};
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def uid() -> str:
    return uuid4().hex


def canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Vault:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.path = self.directory / "workspace.vault"
        self.mutex = threading.RLock()
        self.db: sqlite3.Connection | None = None
        self.key: bytes | None = None
        self.salt: bytes | None = None
        self.last_activity = time.monotonic()
        self.generation = 0
        self.last_migrations: list[str] = []
        self._file_lock = None

    def acquire_process_lock(self) -> None:
        lock_path = self.directory / ".process.lock"
        self._file_lock = open(lock_path, "a+b")
        os.chmod(lock_path, 0o600)
        try:
            fcntl.flock(self._file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file_lock.close()
            self._file_lock = None
            raise RuntimeError("Vault already opened by another server. Use exactly one worker.") from exc

    def close(self) -> None:
        with self.mutex:
            self.lock()
            if self._file_lock:
                fcntl.flock(self._file_lock, fcntl.LOCK_UN)
                self._file_lock.close()
                self._file_lock = None

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def unlocked(self) -> bool:
        return self.db is not None and self.key is not None

    @staticmethod
    def derive(password: str, salt: bytes) -> bytes:
        return Scrypt(salt=salt, length=32, n=2**17, r=8, p=1).derive(password.encode("utf-8"))

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA temp_store=MEMORY")
        return db

    def schema_version(self) -> int:
        if not self.unlocked or self.db is None:
            raise RuntimeError("Vault is locked.")
        return int(self.db.execute("PRAGMA user_version").fetchone()[0])

    @staticmethod
    def _tables(db: sqlite3.Connection) -> set[str]:
        return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    @staticmethod
    def _columns(db: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}

    @staticmethod
    def _require_stage1_shape(db: sqlite3.Connection) -> None:
        required = {"users", "patients", "grants", "documents", "notes", "note_versions", "audit"}
        tables = Vault._tables(db)
        if not required.issubset(tables):
            missing = sorted(required - tables)
            raise UnsupportedSchemaError(
                "The decrypted vault is not the supported Stage 1 schema; "
                f"required tables are missing: {', '.join(missing)}."
            )
        documents_columns = Vault._columns(db, "documents")
        notes_columns = Vault._columns(db, "notes")
        stage2_tables = {
            "sessions", "encounters", "assistant_threads", "assistant_messages",
            "assistant_dialogue_state", "clinical_fact_sources", "clinical_facts",
        }
        if (
            tables & stage2_tables
            or "session_id" in documents_columns
            or "session_id" in notes_columns
            or "encounter_id" in documents_columns
            or "encounter_id" in notes_columns
        ):
            raise UnsupportedSchemaError(
                "The vault contains a partial or different Stage 2 migration. "
                "Restore the untouched Stage 1 backup before applying this release."
            )

    @staticmethod
    def _require_session_shape(db: sqlite3.Connection) -> None:
        tables = Vault._tables(db)
        if not {"sessions", "schema_migrations"}.issubset(tables):
            raise UnsupportedSchemaError(
                "Vault reports schema version 2 but does not contain the session foundation. "
                "Restore the untouched Stage 1 backup rather than using an abandoned Stage 2 vault."
            )
        if "session_id" not in Vault._columns(db, "documents") or "session_id" not in Vault._columns(db, "notes"):
            raise UnsupportedSchemaError(
                "Vault reports schema version 2 but required session links are missing. "
                "Restore the untouched Stage 1 backup."
            )
        migration = db.execute(
            "SELECT description FROM schema_migrations WHERE version=?", (SESSION_FOUNDATION_VERSION,)
        ).fetchone()
        if not migration or migration[0] != SESSION_FOUNDATION_DESCRIPTION:
            raise UnsupportedSchemaError(
                "Vault schema version 2 belongs to a different implementation. "
                "Restore the untouched Stage 1 backup before using this release."
            )
        partial_memory = {
            "assistant_threads", "assistant_messages", "assistant_dialogue_state",
            "clinical_fact_sources", "clinical_facts",
        } & tables
        if partial_memory:
            raise UnsupportedSchemaError(
                "Vault schema version 2 contains a partial later migration. "
                "Restore the verified version-2 backup before retrying."
            )

    @staticmethod
    def _require_assistant_memory_shape(db: sqlite3.Connection) -> None:
        Vault._require_session_shape_for_memory(db)
        tables = Vault._tables(db)
        required = {"assistant_threads", "assistant_messages", "assistant_dialogue_state"}
        if not required.issubset(tables):
            missing = sorted(required - tables)
            raise UnsupportedSchemaError(
                "Vault reports schema version 3 but the encrypted assistant-memory tables are incomplete: "
                + ", ".join(missing)
            )
        thread_columns = Vault._columns(db, "assistant_threads")
        message_columns = Vault._columns(db, "assistant_messages")
        state_columns = Vault._columns(db, "assistant_dialogue_state")
        if not {"user_id", "patient_id", "session_id", "scope", "chart_revision", "version"}.issubset(thread_columns):
            raise UnsupportedSchemaError("Assistant thread schema is incomplete.")
        if not {"thread_id", "sequence", "role", "content", "chart_revision"}.issubset(message_columns):
            raise UnsupportedSchemaError("Assistant message schema is incomplete.")
        if not {"thread_id", "active_intent", "referent_json", "chart_revision"}.issubset(state_columns):
            raise UnsupportedSchemaError("Assistant dialogue-state schema is incomplete.")
        migration = db.execute(
            "SELECT description FROM schema_migrations WHERE version=?", (ASSISTANT_MEMORY_VERSION,)
        ).fetchone()
        if not migration or migration[0] != ASSISTANT_MEMORY_DESCRIPTION:
            raise UnsupportedSchemaError(
                "Vault schema version 3 belongs to a different implementation. "
                "Restore the verified session-foundation backup before using this release."
            )

    @staticmethod
    def _require_session_shape_for_memory(db: sqlite3.Connection) -> None:
        """Validate the version-2 foundation without rejecting version-3 tables."""
        tables = Vault._tables(db)
        if not {"sessions", "schema_migrations"}.issubset(tables):
            raise UnsupportedSchemaError("Assistant memory requires the verified session foundation.")
        if "session_id" not in Vault._columns(db, "documents") or "session_id" not in Vault._columns(db, "notes"):
            raise UnsupportedSchemaError("Assistant memory requires session links on documents and notes.")
        migration = db.execute(
            "SELECT description FROM schema_migrations WHERE version=?", (SESSION_FOUNDATION_VERSION,)
        ).fetchone()
        if not migration or migration[0] != SESSION_FOUNDATION_DESCRIPTION:
            raise UnsupportedSchemaError("The session-foundation migration ledger is missing or incompatible.")

    @staticmethod
    def _create_session_foundation(db: sqlite3.Connection) -> None:
        db.execute(
            """CREATE TABLE sessions (
             id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id),
             title TEXT NOT NULL,
             session_type TEXT NOT NULL CHECK(session_type IN ('office','telehealth','phone','records_review','procedure','other')),
             status TEXT NOT NULL CHECK(status IN ('planned','open','closed','cancelled')),
             starts_at TEXT NOT NULL, ends_at TEXT, location TEXT NOT NULL DEFAULT '',
             clinician_id TEXT REFERENCES users(id), created_by TEXT NOT NULL REFERENCES users(id),
             version INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )"""
        )
        db.execute("CREATE INDEX sessions_patient_idx ON sessions(patient_id, starts_at DESC)")
        db.execute("ALTER TABLE documents ADD COLUMN session_id TEXT REFERENCES sessions(id)")
        db.execute("CREATE INDEX documents_session_idx ON documents(session_id)")
        db.execute("ALTER TABLE notes ADD COLUMN session_id TEXT REFERENCES sessions(id)")
        db.execute("CREATE INDEX notes_session_idx ON notes(session_id)")

    @staticmethod
    def _create_assistant_memory(db: sqlite3.Connection) -> None:
        statements = [
            """CREATE TABLE assistant_threads (
             id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
             patient_id TEXT NOT NULL REFERENCES patients(id), session_id TEXT REFERENCES sessions(id),
             scope TEXT NOT NULL CHECK(scope IN ('longitudinal','session')),
             title TEXT NOT NULL DEFAULT 'Chart conversation',
             status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','archived')),
             chart_revision INTEGER NOT NULL, version INTEGER NOT NULL DEFAULT 1,
             created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
             CHECK((scope='longitudinal' AND session_id IS NULL) OR (scope='session' AND session_id IS NOT NULL))
            )""",
            """CREATE UNIQUE INDEX assistant_threads_longitudinal_unique
             ON assistant_threads(user_id,patient_id) WHERE scope='longitudinal' AND status='active'""",
            """CREATE UNIQUE INDEX assistant_threads_session_unique
             ON assistant_threads(user_id,patient_id,session_id) WHERE scope='session' AND status='active'""",
            "CREATE INDEX assistant_threads_user_patient_idx ON assistant_threads(user_id,patient_id,updated_at DESC)",
            """CREATE TABLE assistant_messages (
             id TEXT PRIMARY KEY, thread_id TEXT NOT NULL REFERENCES assistant_threads(id) ON DELETE CASCADE,
             sequence INTEGER NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant')),
             content TEXT NOT NULL, answer_origin TEXT, source_ids_json TEXT NOT NULL DEFAULT '[]',
             chart_revision INTEGER NOT NULL, created_at TEXT NOT NULL,
             UNIQUE(thread_id,sequence)
            )""",
            "CREATE INDEX assistant_messages_thread_idx ON assistant_messages(thread_id,sequence)",
            """CREATE TABLE assistant_dialogue_state (
             thread_id TEXT PRIMARY KEY REFERENCES assistant_threads(id) ON DELETE CASCADE,
             active_intent TEXT, referent_json TEXT NOT NULL DEFAULT '{}',
             requested_format_json TEXT NOT NULL DEFAULT '{}',
             safety_mode TEXT NOT NULL DEFAULT 'chart_only',
             last_source_ids_json TEXT NOT NULL DEFAULT '[]',
             chart_revision INTEGER NOT NULL, updated_at TEXT NOT NULL
            )""",
        ]
        for statement in statements:
            db.execute(statement)

    @staticmethod
    def _require_fact_store_shape(db: sqlite3.Connection) -> None:
        """Validate the schema-version-4 provenance store before v5 upgrade."""
        Vault._require_assistant_memory_shape(db)
        tables = Vault._tables(db)
        required = {"clinical_fact_sources", "clinical_facts"}
        if not required.issubset(tables):
            raise UnsupportedSchemaError(
                "Vault reports schema version 4 but provenance fact tables are incomplete: "
                + ", ".join(sorted(required - tables))
            )
        source_columns = Vault._columns(db, "clinical_fact_sources")
        fact_columns = Vault._columns(db, "clinical_facts")
        if not {"source_kind","source_id","patient_id","source_version","source_sha256","extraction_version","fact_count"}.issubset(source_columns):
            raise UnsupportedSchemaError("Provenance fact-source schema is incomplete.")
        if not {"patient_id","fact_type","entity_key","event_type","source_kind","source_id","source_version","source_sha256","quote","start_offset","end_offset","extraction_version"}.issubset(fact_columns):
            raise UnsupportedSchemaError("Provenance fact schema is incomplete.")
        migration = db.execute(
            "SELECT description FROM schema_migrations WHERE version=?", (PROVENANCE_FACT_VERSION,)
        ).fetchone()
        if not migration or migration[0] != PROVENANCE_FACT_DESCRIPTION:
            raise UnsupportedSchemaError(
                "Vault schema version 4 belongs to a different implementation. Restore the verified version-3 backup."
            )

    @staticmethod
    def _require_fact_semantics_shape(db: sqlite3.Connection) -> None:
        Vault._require_fact_store_shape(db)
        tables = Vault._tables(db)
        if "clinical_fact_resolutions" not in tables:
            raise UnsupportedSchemaError("Vault reports schema version 5 but the fact-resolution table is missing.")
        fact_columns = Vault._columns(db, "clinical_facts")
        required_fact_columns = {
            "event_role", "asserted_at", "effective_start", "effective_end",
            "assertion_key", "supersedes_json", "semantic_version",
        }
        if not required_fact_columns.issubset(fact_columns):
            raise UnsupportedSchemaError(
                "Fact-semantics columns are incomplete: "
                + ", ".join(sorted(required_fact_columns - fact_columns))
            )
        resolution_columns = Vault._columns(db, "clinical_fact_resolutions")
        if not {
            "patient_id", "resolution_key", "fact_type", "entity_key", "event_type",
            "event_role", "status", "preferred_fact_id", "fact_ids_json",
            "superseded_fact_ids_json", "values_json",
        }.issubset(resolution_columns):
            raise UnsupportedSchemaError("Fact-resolution schema is incomplete.")
        migration = db.execute(
            "SELECT description FROM schema_migrations WHERE version=?", (SCHEMA_VERSION,)
        ).fetchone()
        if not migration or migration[0] != FACT_SEMANTICS_DESCRIPTION:
            raise UnsupportedSchemaError(
                "Vault schema version 5 belongs to a different implementation. Restore the verified version-4 backup."
            )

    @staticmethod
    def _create_fact_store(db: sqlite3.Connection) -> None:
        statements = [
            """CREATE TABLE clinical_fact_sources (
             source_kind TEXT NOT NULL CHECK(source_kind IN ('document','note')),
             source_id TEXT NOT NULL, patient_id TEXT NOT NULL REFERENCES patients(id),
             source_version INTEGER NOT NULL, source_sha256 TEXT NOT NULL,
             extraction_version TEXT NOT NULL, fact_count INTEGER NOT NULL, indexed_at TEXT NOT NULL,
             PRIMARY KEY(source_kind,source_id)
            )""",
            "CREATE INDEX clinical_fact_sources_patient_idx ON clinical_fact_sources(patient_id)",
            """CREATE TABLE clinical_facts (
             id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id), session_id TEXT REFERENCES sessions(id),
             fact_type TEXT NOT NULL, entity_key TEXT, entity_label TEXT, event_type TEXT,
             value_text TEXT NOT NULL, normalized_json TEXT NOT NULL DEFAULT '{}', event_date TEXT,
             status TEXT NOT NULL DEFAULT 'current', polarity TEXT NOT NULL DEFAULT 'affirmed',
             certainty TEXT NOT NULL DEFAULT 'explicit', source_kind TEXT NOT NULL CHECK(source_kind IN ('document','note')),
             source_id TEXT NOT NULL, source_version INTEGER NOT NULL, source_sha256 TEXT NOT NULL,
             source_title TEXT NOT NULL, quote TEXT NOT NULL, start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL,
             extraction_method TEXT NOT NULL, extraction_version TEXT NOT NULL, created_at TEXT NOT NULL,
             UNIQUE(source_kind,source_id,source_version,fact_type,entity_key,event_type,start_offset,end_offset,value_text)
            )""",
            "CREATE INDEX clinical_facts_patient_type_idx ON clinical_facts(patient_id,fact_type)",
            "CREATE INDEX clinical_facts_patient_entity_idx ON clinical_facts(patient_id,entity_key)",
            "CREATE INDEX clinical_facts_patient_event_idx ON clinical_facts(patient_id,event_type,event_date)",
            "CREATE INDEX clinical_facts_source_idx ON clinical_facts(source_kind,source_id,source_version)",
        ]
        for statement in statements:
            db.execute(statement)

    @staticmethod
    def _upgrade_fact_semantics(db: sqlite3.Connection) -> None:
        columns = Vault._columns(db, "clinical_facts")
        additions = [
            ("event_role", "TEXT NOT NULL DEFAULT 'unspecified'"),
            ("asserted_at", "TEXT"),
            ("effective_start", "TEXT"),
            ("effective_end", "TEXT"),
            ("assertion_key", "TEXT NOT NULL DEFAULT ''"),
            ("supersedes_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("semantic_version", "TEXT NOT NULL DEFAULT 'fact-semantics-v1'"),
        ]
        for column, ddl in additions:
            if column not in columns:
                db.execute(f"ALTER TABLE clinical_facts ADD COLUMN {column} {ddl}")
        db.execute(
            "CREATE INDEX IF NOT EXISTS clinical_facts_assertion_idx "
            "ON clinical_facts(patient_id,assertion_key)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS clinical_facts_patient_event_role_idx "
            "ON clinical_facts(patient_id,event_type,event_role,event_date)"
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS clinical_fact_resolutions (
             id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id),
             resolution_key TEXT NOT NULL, fact_type TEXT NOT NULL, entity_key TEXT,
             event_type TEXT, event_role TEXT NOT NULL DEFAULT 'unspecified', status TEXT NOT NULL,
             preferred_fact_id TEXT, fact_ids_json TEXT NOT NULL DEFAULT '[]',
             superseded_fact_ids_json TEXT NOT NULL DEFAULT '[]', values_json TEXT NOT NULL DEFAULT '{}',
             computed_at TEXT NOT NULL, UNIQUE(patient_id,resolution_key)
            )"""
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS clinical_fact_resolutions_patient_idx "
            "ON clinical_fact_resolutions(patient_id,fact_type,status)"
        )
        # v4 facts are a rebuildable derived cache. Clearing them avoids mixing
        # old extraction semantics with v5 conflict/lifecycle semantics.
        db.execute("DELETE FROM clinical_facts")
        db.execute("DELETE FROM clinical_fact_sources")
        db.execute("DELETE FROM clinical_fact_resolutions")

    def _migrate_if_needed(self) -> list[str]:
        if not self.unlocked or self.db is None:
            raise RuntimeError("Vault is locked.")
        version = self.schema_version()
        if version == SCHEMA_VERSION:
            self._require_fact_semantics_shape(self.db)
            return []
        if version not in {0, 1, SESSION_FOUNDATION_VERSION, ASSISTANT_MEMORY_VERSION, PROVENANCE_FACT_VERSION}:
            raise UnsupportedSchemaError(f"Vault schema version {version} is not supported by this release.")
        if version in {0, 1}:
            self._require_stage1_shape(self.db)
        elif version == SESSION_FOUNDATION_VERSION:
            self._require_session_shape(self.db)
        elif version == ASSISTANT_MEMORY_VERSION:
            self._require_assistant_memory_shape(self.db)
            partial_facts = {"clinical_fact_sources", "clinical_facts"} & self._tables(self.db)
            if partial_facts:
                raise UnsupportedSchemaError(
                    "Vault schema version 3 contains a partial provenance-fact migration. "
                    "Restore the verified version-3 backup before retrying."
                )
        else:
            self._require_fact_store_shape(self.db)

        applied: list[str] = []
        original_version = version
        with self.transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL, description TEXT NOT NULL)"
            )
            db.execute(
                "INSERT OR IGNORE INTO schema_migrations(version,applied_at,description) VALUES(?,?,?)",
                (1, now(), "Stage 1 records foundation"),
            )
            if version < SESSION_FOUNDATION_VERSION:
                self._create_session_foundation(db)
                db.execute(
                    "INSERT INTO schema_migrations(version,applied_at,description) VALUES(?,?,?)",
                    (SESSION_FOUNDATION_VERSION, now(), SESSION_FOUNDATION_DESCRIPTION),
                )
                self.audit("system", None, "schema.migrated", "schema", {
                    "from_version": original_version, "to_version": SESSION_FOUNDATION_VERSION,
                    "description": SESSION_FOUNDATION_DESCRIPTION,
                    "existing_documents_session_id": None, "existing_notes_session_id": None,
                })
                applied.append(f"Migrated Stage 1 vault to schema version {SESSION_FOUNDATION_VERSION}: {SESSION_FOUNDATION_DESCRIPTION}")
                version = SESSION_FOUNDATION_VERSION
            if version < ASSISTANT_MEMORY_VERSION:
                self._create_assistant_memory(db)
                db.execute(
                    "INSERT INTO schema_migrations(version,applied_at,description) VALUES(?,?,?)",
                    (ASSISTANT_MEMORY_VERSION, now(), ASSISTANT_MEMORY_DESCRIPTION),
                )
                self.audit("system", None, "schema.migrated", "schema", {
                    "from_version": version, "to_version": ASSISTANT_MEMORY_VERSION,
                    "description": ASSISTANT_MEMORY_DESCRIPTION, "existing_threads": 0,
                })
                applied.append(f"Migrated vault to schema version {ASSISTANT_MEMORY_VERSION}: {ASSISTANT_MEMORY_DESCRIPTION}")
                version = ASSISTANT_MEMORY_VERSION
            if version < PROVENANCE_FACT_VERSION:
                self._create_fact_store(db)
                db.execute(
                    "INSERT INTO schema_migrations(version,applied_at,description) VALUES(?,?,?)",
                    (PROVENANCE_FACT_VERSION, now(), PROVENANCE_FACT_DESCRIPTION),
                )
                self.audit("system", None, "schema.migrated", "schema", {
                    "from_version": version, "to_version": PROVENANCE_FACT_VERSION,
                    "description": PROVENANCE_FACT_DESCRIPTION, "derived_fact_rows": 0,
                })
                applied.append(f"Migrated vault to schema version {PROVENANCE_FACT_VERSION}: {PROVENANCE_FACT_DESCRIPTION}")
                version = PROVENANCE_FACT_VERSION
            if version < SCHEMA_VERSION:
                prior_fact_rows = db.execute("SELECT COUNT(*) FROM clinical_facts").fetchone()[0]
                self._upgrade_fact_semantics(db)
                db.execute(
                    "INSERT INTO schema_migrations(version,applied_at,description) VALUES(?,?,?)",
                    (SCHEMA_VERSION, now(), FACT_SEMANTICS_DESCRIPTION),
                )
                db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                self.audit("system", None, "schema.migrated", "schema", {
                    "from_version": version, "to_version": SCHEMA_VERSION,
                    "description": FACT_SEMANTICS_DESCRIPTION,
                    "derived_fact_rows_discarded_for_rebuild": prior_fact_rows,
                })
                applied.append(f"Migrated vault to schema version {SCHEMA_VERSION}: {FACT_SEMANTICS_DESCRIPTION}")
        self._require_fact_semantics_shape(self.db)
        return applied

    def create(self, password: str) -> None:
        with self.mutex:
            if self.exists:
                raise ValueError("A vault already exists.")
            self.salt = os.urandom(16)
            self.key = self.derive(password, self.salt)
            self.db = self._connect()
            self.db.executescript(SCHEMA)
            self.generation += 1
            self.last_activity = time.monotonic()
            self.last_migrations = [
                f"Created schema version {SCHEMA_VERSION}: {FACT_SEMANTICS_DESCRIPTION}"
            ]
            # The first save happens in the administrator-creation transaction.

    def unlock(self, password: str) -> None:
        with self.mutex:
            if self.unlocked:
                raise ValueError("Vault is already unlocked. Sign in with your own account.")
            raw = self.path.read_bytes()
            if not raw.startswith(MAGIC) or len(raw) < len(MAGIC) + 44:
                raise ValueError("Invalid vault format.")
            offset = len(MAGIC)
            salt, nonce, ciphertext = raw[offset:offset+16], raw[offset+16:offset+28], raw[offset+28:]
            key = self.derive(password, salt)
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, MAGIC + salt)
            db = self._connect()
            try:
                db.deserialize(plaintext)
                db.execute("PRAGMA foreign_keys=ON")
                db.execute("PRAGMA temp_store=MEMORY")
                if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("Database integrity check failed.")
                self.db, self.key, self.salt = db, key, salt
                if not self.verify_audit()["valid"]:
                    raise ValueError("Audit integrity check failed.")
                self.last_migrations = self._migrate_if_needed()
            except Exception:
                db.close()
                self.db = None
                self.key = None
                self.salt = None
                self.last_migrations = []
                raise
            self.generation += 1
            self.last_activity = time.monotonic()

    def lock(self) -> None:
        with self.mutex:
            if self.db:
                self.db.close()
            self.db = None
            self.key = None
            self.salt = None
            self.last_migrations = []
            self.generation += 1
            # CPython cannot guarantee zeroization of every plaintext copy.

    def persist(self) -> None:
        if not self.unlocked or self.db is None or self.key is None or self.salt is None:
            raise RuntimeError("Vault is locked.")
        serialized = self.db.serialize()
        if len(serialized) > MAX_DB_BYTES:
            raise ValueError("Prototype vault limit reached (96 MiB). Export an encrypted backup.")
        nonce = os.urandom(12)
        payload = MAGIC + self.salt + nonce + AESGCM(self.key).encrypt(nonce, serialized, MAGIC + self.salt)
        temp = self.directory / (".save-" + uid())
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp, self.path)
            try:
                dir_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        finally:
            temp.unlink(missing_ok=True)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.mutex:
            if not self.unlocked or self.db is None:
                raise RuntimeError("Vault is locked.")
            previous = self.db.serialize()
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
                self.db.execute("COMMIT")
                self.persist()
            except Exception:
                if self.db.in_transaction:
                    self.db.execute("ROLLBACK")
                self.db.deserialize(previous)
                self.db.execute("PRAGMA foreign_keys=ON")
                raise

    def audit(self, actor: str, patient_id: str | None, action: str, entity_id: str, details: dict) -> None:
        if self.db is None or self.key is None:
            raise RuntimeError("Vault is locked.")
        last = self.db.execute("SELECT event_hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        previous = last[0] if last else "0" * 64
        fields = [now(), actor, patient_id, action, entity_id, canonical(details), previous]
        audit_key = hmac.new(self.key, b"clinical-workspace/audit/v1", hashlib.sha256).digest()
        digest = hmac.new(audit_key, canonical(fields).encode(), hashlib.sha256).hexdigest()
        self.db.execute(
            "INSERT INTO audit(at,actor,patient_id,action,entity_id,details,previous_hash,event_hash) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (*fields, digest),
        )

    def verify_audit(self) -> dict:
        if self.db is None or self.key is None:
            raise RuntimeError("Vault is locked.")
        previous = "0" * 64
        count = 0
        audit_key = hmac.new(self.key, b"clinical-workspace/audit/v1", hashlib.sha256).digest()
        for row in self.db.execute("SELECT * FROM audit ORDER BY seq"):
            fields = [row[key] for key in [
                "at", "actor", "patient_id", "action", "entity_id", "details", "previous_hash"
            ]]
            digest = hmac.new(audit_key, canonical(fields).encode(), hashlib.sha256).hexdigest()
            if row["previous_hash"] != previous or not hmac.compare_digest(digest, row["event_hash"]):
                return {"valid": False, "checked_events": count, "broken_at": row["seq"]}
            previous = row["event_hash"]
            count += 1
        return {
            "valid": True,
            "checked_events": count,
            "head": previous,
            "limitation": "Local chain only; no external anchor. Cannot detect whole-vault rollback or privileged rewriting.",
        }
