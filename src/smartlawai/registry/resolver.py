"""Resolve raw citation string to registry target. Re-exported in smartlawai."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from registry.resolver import resolve_citation

__all__ = ["resolve_citation"]
