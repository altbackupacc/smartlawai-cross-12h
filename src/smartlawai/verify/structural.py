"""Structural verification: every claim carries >= 1 valid passage_id.

Implements M5.1:
- Every claim must carry at least one passage_id (I12 / PLAN.md M5.1).
- Cited passage_ids must exist in the retrieved passages.
- Fails closed on missing or ungroundable structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smartlawai.adapters.base import RetrievedChunk
from smartlawai.protocols import Claim


@dataclass
class StructuralResult:
    """Per-claim structural verification outcome."""

    claim_index: int
    is_valid: bool
    passage_ids: list[str] = field(default_factory=list)
    status: str = "ok"  # "ok" | "failed" | "unavailable"
    reason: str = ""


def verify_structural_claim(
    claim: Claim, valid_passage_ids: set[str], claim_index: int = 0
) -> StructuralResult:
    """Verify a single claim's structural provenance."""
    if not claim.passage_ids:
        return StructuralResult(
            claim_index=claim_index,
            is_valid=False,
            passage_ids=[],
            status="failed",
            reason="missing_passage_id",
        )

    # Check if all cited passage_ids are in the retrieved pool
    missing = [pid for pid in claim.passage_ids if pid not in valid_passage_ids]
    if missing:
        return StructuralResult(
            claim_index=claim_index,
            is_valid=False,
            passage_ids=claim.passage_ids,
            status="failed",
            reason=f"invalid_passage_ids: {missing}",
        )

    return StructuralResult(
        claim_index=claim_index,
        is_valid=True,
        passage_ids=claim.passage_ids,
        status="ok",
        reason="valid_provenance",
    )


def verify_structural(
    claims: list[Claim], passages: list[RetrievedChunk]
) -> list[StructuralResult]:
    """Verify all claims against the retrieved passage pool."""
    valid_ids = {
        p.chunk.chunk_id
        for p in passages
        if p and p.chunk and p.chunk.chunk_id
    }

    results: list[StructuralResult] = []
    for idx, claim in enumerate(claims):
        results.append(verify_structural_claim(claim, valid_ids, claim_index=idx))
    return results
