"""Authority check: was this citation in force on a given date?

Public contract consumed by gate.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verify.registry_check import check_citation

__all__ = ["check_citation"]
