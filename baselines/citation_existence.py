"""Citation-existence-only baseline: flags a citation as valid iff it exists
somewhere in the corpus, with NO in-force/date dimension at all. Small-scale
replication of arXiv 2606.00898's citation-graph mechanism (PLAN.md M7).

This deliberately reproduces that paper's reported gap on this project's own
corpus: a citation to a repealed IPC section will resolve exists_in_corpus=True,
because the pre-2024 statutory text is still verbatim throughout the corpus --
existence-checking has no way to know the provision was superseded. A real
registry check (M4, not built) would additionally resolve the citation against
Gazette/eSCR-sourced in-force intervals. Run this against eval/gold/repealed.jsonl
once that gold set exists to make the gap concrete rather than asserted.

Reuses core/ner.py's extraction (no new citation-parsing logic)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from smartlawai.adapters.base import BackendInterface
from smartlawai.core.ner import extract_entities
from smartlawai.scope import Scope

_WS = re.compile(r"\s+")


def _normalize(citation: str) -> str:
    return _WS.sub(" ", citation).strip().lower()


@dataclass
class CitationExistenceResult:
    citation: str
    status: str  # "ok" | "unavailable" -- I2-style discipline for a missing index
    exists_in_corpus: bool | None
    reason: str = ""


def build_corpus_citation_index(backend: BackendInterface, scope: Scope) -> set[str]:
    """I1: scoped to the eval run's doc_ids, never fetch_all_chunks(). Extracts
    statutes + case_citations + legal_provisions from every in-scope chunk."""
    chunks = backend.fetch_chunks_scoped(scope)
    index: set[str] = set()
    for c in chunks:
        r = extract_entities(c.doc_id, c.chunk_text)
        index.update(_normalize(x) for x in
                     r.statutes + r.case_citations + r.legal_provisions)
    return index


def check_existence(citation: str, index: set[str] | None) -> CitationExistenceResult:
    if index is None:
        return CitationExistenceResult(citation, status="unavailable",
                                       exists_in_corpus=None,
                                       reason="corpus_index_unavailable")
    return CitationExistenceResult(citation, status="ok",
                                   exists_in_corpus=_normalize(citation) in index)


def to_registry_result_dict(result: CitationExistenceResult) -> dict:
    """Real shape now that M4/M6 have landed (this note replaces an earlier
    one written when both were still specs): eval/metrics.py's
    citation_validity_rate/registry_coverage_rate read a `"status"` key with
    one of M4's four RegistryResult values (registry/types.py) -- not the
    `{"valid", "in_force"}` shape this function originally guessed, which
    doesn't match and would make those metrics report `unavailable:
    malformed_registry_status`.

    This baseline structurally cannot determine WHICH resolved status
    applies to an existing citation -- that is the entire point of the
    comparison PLAN.md wants (existence-checking has no temporal
    dimension). citation_validity_rate/registry_coverage_rate only care
    whether a citation resolved to *something* real, and treat in_force/
    repealed/superseded identically for that purpose, so mapping an
    existing citation to "in_force" is safe for those two metrics. It is
    NOT safe input for repealed_citation_rate, which specifically needs to
    distinguish repealed from in_force -- feeding this baseline's output
    into that metric would silently fabricate a status this method has no
    basis for (the exact I2 failure mode M4/M6 were both built to avoid).
    Use M4's real verify/registry_check.py for repealed_citation_rate
    instead; the demonstrated gap *is* that this baseline can only ever
    answer "exists somewhere in the text", never "is still valid law"."""
    if result.status == "unavailable":
        return {"status": "not_found", "as_of": None, "note": result.reason}
    return {
        "status": "in_force" if result.exists_in_corpus else "not_found",
        "as_of": None,
        "note": ("citation-existence-only baseline: no in-force/date check "
                 "performed" if result.exists_in_corpus else None),
    }
