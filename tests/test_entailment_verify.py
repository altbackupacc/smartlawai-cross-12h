"""Tests for entailment verifier dual-scorer architecture (Invariant I2 & I11)."""

from __future__ import annotations

from smartlawai.adapters.base import Chunk, RetrievedChunk
from smartlawai.protocols import Claim
from smartlawai.verify.entailment import (
    DualEntailmentVerifier,
    EntailmentScore,
    HHEMScorer,
)


class MockHHEM(HHEMScorer):
    def __init__(self, score_val: float | None = 0.9, status: str = "ok"):
        super().__init__()
        self._val = score_val
        self._status = status

    def score_pair(self, passage: str, claim: str) -> EntailmentScore:
        return EntailmentScore(name="hhem-2.1", value=self._val, status=self._status, reason="mock")


def test_dual_entailment_verifier_records_both_scorers():
    """Invariant I11: Both HHEM and InLegalNLI must be recorded."""
    mock_hhem = MockHHEM(score_val=0.92, status="ok")
    verifier = DualEntailmentVerifier(hhem_scorer=mock_hhem)

    claim = Claim(text="The tenant breached clause 4.", passage_ids=["p1"])
    passages = {
        "p1": RetrievedChunk(
            chunk=Chunk(
                chunk_id="p1",
                doc_id="d1",
                chunk_index=0,
                chunk_text="Tenant committed breach of clause 4.",
                char_start=0,
                char_end=35,
            ),
            score=1.0,
        )
    }

    res = verifier.verify_claim(claim, passages)
    assert res.hhem_score.name == "hhem-2.1"
    assert res.hhem_score.value == 0.92
    assert res.inlegalnli_score.name == "inlegalnli"
    assert res.inlegalnli_score.status == "unavailable"  # Pending M8a
    assert res.is_entailed is True
    assert res.status == "ok"


def test_dual_entailment_invariant_i2_fails_closed_when_hhem_unavailable():
    """Invariant I2: If HHEM cannot run, status must be unavailable, never a fake score."""
    mock_hhem = MockHHEM(score_val=None, status="unavailable")
    verifier = DualEntailmentVerifier(hhem_scorer=mock_hhem)

    claim = Claim(text="Some statement.", passage_ids=["p1"])
    passages = {
        "p1": RetrievedChunk(
            chunk=Chunk(
                chunk_id="p1",
                doc_id="d1",
                chunk_index=0,
                chunk_text="Some passage.",
                char_start=0,
                char_end=13,
            ),
            score=1.0,
        )
    }

    res = verifier.verify_claim(claim, passages)
    assert res.status == "unavailable"
    assert res.is_entailed is False
    assert res.hhem_score.status == "unavailable"
    assert res.hhem_score.value is None  # Never a 0.5 default!


def test_dual_entailment_claim_missing_passage_fails_closed():
    verifier = DualEntailmentVerifier(hhem_scorer=MockHHEM())
    claim = Claim(text="Uncited claim.", passage_ids=[])
    res = verifier.verify_claim(claim, {})
    assert res.status == "unavailable"
    assert res.is_entailed is False
