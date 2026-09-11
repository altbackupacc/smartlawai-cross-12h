"""Tests for structural verification (PLAN.md M5.1)."""

from __future__ import annotations

from smartlawai.adapters.base import Chunk, RetrievedChunk
from smartlawai.protocols import Claim
from smartlawai.verify.structural import verify_structural, verify_structural_claim


def test_claim_without_passage_id_fails_structural_verification():
    claim = Claim(text="Claim with no citations.", passage_ids=[])
    res = verify_structural_claim(claim, valid_passage_ids={"p1", "p2"})
    assert not res.is_valid
    assert res.status == "failed"
    assert res.reason == "missing_passage_id"


def test_claim_with_invalid_passage_id_fails_structural_verification():
    claim = Claim(text="Claim citing unknown chunk.", passage_ids=["chunk_999"])
    res = verify_structural_claim(claim, valid_passage_ids={"p1", "p2"})
    assert not res.is_valid
    assert res.status == "failed"
    assert "invalid_passage_ids" in res.reason


def test_claim_with_valid_passage_id_passes():
    claim = Claim(text="Well cited claim.", passage_ids=["p1"])
    res = verify_structural_claim(claim, valid_passage_ids={"p1", "p2"})
    assert res.is_valid
    assert res.status == "ok"


def test_batch_verify_structural():
    c1 = Claim(text="First claim.", passage_ids=["p1"])
    c2 = Claim(text="Second claim.", passage_ids=["missing"])
    c3 = Claim(text="Third claim.", passage_ids=[])

    chunks = [
        RetrievedChunk(
            chunk=Chunk(chunk_id="p1", doc_id="d1", chunk_index=0, chunk_text="Text 1", char_start=0, char_end=6),
            score=1.0,
        ),
        RetrievedChunk(
            chunk=Chunk(chunk_id="p2", doc_id="d1", chunk_index=1, chunk_text="Text 2", char_start=7, char_end=13),
            score=0.8,
        ),
    ]

    results = verify_structural([c1, c2, c3], chunks)
    assert len(results) == 3
    assert results[0].is_valid is True
    assert results[1].is_valid is False
    assert results[2].is_valid is False
