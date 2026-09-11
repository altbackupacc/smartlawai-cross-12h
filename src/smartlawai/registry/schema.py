"""The authority registry database wrapper. Re-exported in smartlawai."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from registry.schema import DEFAULT_DB, RegistryDB

__all__ = ["DEFAULT_DB", "RegistryDB"]
