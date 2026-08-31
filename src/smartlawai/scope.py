"""I1: every retrieval call takes an explicit Scope. There is no default and no
'search everything' path. See CLAUDE.md I1."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scope:
    doc_ids: tuple[str, ...]
    owner_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_ids", tuple(self.doc_ids))
        if not self.doc_ids:
            raise ValueError("Scope.doc_ids must be non-empty (I1: no 'search everything')")
        if not self.owner_id:
            raise ValueError("Scope.owner_id is required (I1)")

    def contains(self, doc_id: str) -> bool:
        return doc_id in self.doc_ids
