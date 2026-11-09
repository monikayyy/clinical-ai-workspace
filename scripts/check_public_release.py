#!/usr/bin/env python3
"""Fail if a public portfolio checkout contains obvious private/runtime artifacts.

This is intentionally conservative and complements (not replaces) a real secret scanner.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DISALLOWED_SUFFIXES = {
    ".vault", ".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx"
}
DISALLOWED_NAMES = {".env", "id_rsa", "id_ed25519"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
TEXT_LIMIT = 2 * 1024 * 1024

PATTERNS = [
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b")),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
]

# Known synthetic/dev literals are deliberately permitted. These are not secrets.
ALLOW_TEXT = {
    "local-development",
    "synthetic-key",
    "EXACT SECRET SENTINEL",
    "SECRET FOREIGN SENTINEL",
}


def iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path


def main() -> int:
    failures: list[str] = []
    for path in iter_files():
        rel = path.relative_to(ROOT)
        if path.name in DISALLOWED_NAMES or path.suffix.lower() in DISALLOWED_SUFFIXES:
            failures.append(f"disallowed artifact: {rel}")
            continue
        if path.stat().st_size > TEXT_LIMIT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        sanitized = text
        for allowed in ALLOW_TEXT:
            sanitized = sanitized.replace(allowed, "")
        for label, pattern in PATTERNS:
            if pattern.search(sanitized):
                failures.append(f"credential-like content ({label}): {rel}")

    if failures:
        print("PUBLIC_RELEASE_CHECK_FAIL")
        for item in failures:
            print(f"- {item}")
        return 1
    print("PUBLIC_RELEASE_CHECK_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
