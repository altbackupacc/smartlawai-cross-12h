"""Normalise raw citation text to canonical registry identifiers. Re-exported in smartlawai."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from registry.normalise import (
    ACTS,
    abbrev,
    is_compatible,
    normalise_act,
    parse_section_ref,
    section_id,
)

__all__ = [
    "ACTS",
    "abbrev",
    "is_compatible",
    "normalise_act",
    "parse_section_ref",
    "section_id",
]
