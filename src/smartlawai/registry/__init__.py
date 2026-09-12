"""SmartLawAI Authority Registry module."""

from __future__ import annotations

from smartlawai.registry.normalise import is_compatible, normalise_act, parse_section_ref
from smartlawai.registry.resolver import resolve_citation
from smartlawai.registry.schema import DEFAULT_DB, RegistryDB
from smartlawai.registry.types import (
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
    "DEFAULT_DB",
    "STATUS_IN_FORCE",
    "STATUS_NOT_FOUND",
    "STATUS_REPEALED",
    "STATUS_SUPERSEDED",
    "TARGET_SECTION",
    "TARGET_STATUTE",
    "RegistryDB",
    "RegistryResult",
    "ResolvedCitation",
    "is_compatible",
    "normalise_act",
    "parse_section_ref",
    "resolve_citation",
]
