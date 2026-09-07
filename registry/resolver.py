"""Resolve a raw citation string to a registry target.

Public contract (frozen -- M4_ONBOARDING.md 6, consumed by M5's `gate.py`):

    resolve_citation(raw_text: str) -> ResolvedCitation | None

Returns None when the citation cannot be resolved *to a row that exists in the
registry*. That is the honest answer, and it is what feeds the unresolved count
in coverage reporting. It never guesses a target (CLAUDE.md I2).

Resolution is deliberately layered, most-certain first:

1. Act + section both present and the section exists  -> section, confidence 1.0
2. Act recognised, no section given, act exists       -> statute, confidence 0.9
3. Anything else                                      -> None
"""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb

from registry.normalise import is_compatible, normalise_act, parse_section_ref, section_id
from registry.schema import DEFAULT_DB
from registry.types import TARGET_SECTION, TARGET_STATUTE, ResolvedCitation

_conn = None
_lock = threading.Lock()


def _db(path: str | Path | None = None):
    """Lazily open a read-only connection.

    Read-only so that a serving process resolving citations can never mutate
    the registry, and so several readers can share the file.
    """
    global _conn
    if path is not None:
        return duckdb.connect(str(path), read_only=True)
    with _lock:
        if _conn is None:
            _conn = duckdb.connect(str(DEFAULT_DB), read_only=True)
        return _conn


def resolve_citation(raw_text: str, db=None) -> ResolvedCitation | None:
    """Normalise a raw citation to a registry target, or None if it cannot be.

    `raw_text` is what `smartlawai.core.ner.extract_entities()` emits as a
    STATUTE or PROVISION entity, e.g. "Section 420 IPC".
    """
    if not raw_text or not raw_text.strip():
        return None

    conn = db if db is not None else _db()
    act = normalise_act(raw_text)
    if act is None:
        return None

    ref = parse_section_ref(raw_text)

    # --- 1. act + section ------------------------------------------------
    if ref is not None:
        if not is_compatible(act, ref.kind):
            # e.g. "Article 21 IPC" -- structurally impossible, do not invent.
            return None
        sid = section_id(act, ref)
        hit = conn.execute(
            "SELECT id FROM SECTIONS WHERE id = ?", [sid]
        ).fetchone()
        if hit:
            return ResolvedCitation(
                raw=raw_text.strip(), target_type=TARGET_SECTION,
                target_id=sid, confidence=1.0,
            )
        # Parsed cleanly but no such row. Unresolved, not guessed.
        return None

    # --- 2. act only -----------------------------------------------------
    hit = conn.execute("SELECT id FROM STATUTES WHERE id = ?", [act]).fetchone()
    if hit:
        return ResolvedCitation(
            raw=raw_text.strip(), target_type=TARGET_STATUTE,
            target_id=act, confidence=0.9,
        )
    return None
