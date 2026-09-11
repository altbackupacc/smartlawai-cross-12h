"""Frozen interface contract for authority registry. Re-exported in smartlawai."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root registry is accessible
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from registry.types import (
    STATUS_IN_FORCE,
    STATUS_NOT_FOUND,
    STATUS_REPEALED,
    STATUS_SUPERSEDED,
    TARGET_SECTION,
    TARGET_STATUTE,
    RegistryResult,
    ResolvedCitation,
)

__all__ = [
    "STATUS_IN_FORCE",
    "STATUS_NOT_FOUND",
    "STATUS_REPEALED",
    "STATUS_SUPERSEDED",
    "TARGET_SECTION",
    "TARGET_STATUTE",
    "RegistryResult",
    "ResolvedCitation",
]
