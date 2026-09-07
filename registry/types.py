"""Frozen interface contract between M4 (registry) and M5 (gate).

`OPS.md` §8 names `Citation` -- what `ner.py` emits, what `registry` consumes --
as one of the contracts to freeze on day one. `M4_ONBOARDING.md` §6 gives the
exact shape. Nothing here may change without telling whoever owns `gate.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Status values returned by check_citation(). "not_found" is an honest answer,
# never a fallback -- see CLAUDE.md I2.
STATUS_IN_FORCE = "in_force"
STATUS_REPEALED = "repealed"
STATUS_SUPERSEDED = "superseded"
STATUS_NOT_FOUND = "not_found"

TARGET_SECTION = "section"
TARGET_STATUTE = "statute"


@dataclass
class ResolvedCitation:
    """A raw citation string normalised to a registry target.

    Produced by `resolver.resolve_citation()` from the `STATUTE` / `PROVISION`
    entities that `smartlawai.core.ner.extract_entities()` emits.
    """

    raw: str
    target_type: str        # TARGET_SECTION | TARGET_STATUTE
    target_id: str
    confidence: float


@dataclass
class RegistryResult:
    """The authority verdict on a citation, as of a date.

    `gate.py` treats every non-`in_force` status the same way: it strikes the
    claim. It never passes a claim through on a guess (CLAUDE.md I2, I3).
    """

    status: str              # STATUS_* above
    as_of: date
    note: str | None = None  # e.g. "repealed 2024-07-01; see BNS 318"
