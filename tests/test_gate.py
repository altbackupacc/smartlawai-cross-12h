"""Tests for deterministic gate decision engine (Invariant I2, I3, I12)."""

from __future__ import annotations

from datetime import date

from smartlawai.gate import GateOutcome, decide_gate
from smartlawai.protocols import Claim
from smartlawai.verify.entailment import ClaimEntailmentResult, EntailmentScore
from smartlawai.verify.structural import StructuralResult

TODAY = date(2026, 9, 10)
PAST_DATE = date(2020, 1, 1)


def make_mock_entailment(claim_text: str, is_entailed: bool = True, status: str = "ok") -> ClaimEntailmentResult:
    return ClaimEntailmentResult(
        claim_text=claim_text,
        passage_id="p1",
        hhem_score=EntailmentScore(name="hhem", value=0.9 if is_entailed else 0.1, status=status),
        inlegalnli_score=EntailmentScore(name="inlegalnli", value=None, status="unavailable"),
        is_entailed=is_entailed,
        status=status,
    )


def test_gate_allows_when_all_components_pass():
    claim = Claim(text="The agreement was signed in New Delhi.", passage_ids=["p1"], citations=[])
    struct = [StructuralResult(claim_index=0, is_valid=True, passage_ids=["p1"], status="ok")]
    entail = [make_mock_entailment(claim.text, is_entailed=True)]

    decision = decide_gate([claim], struct, entail, as_of=TODAY)
    assert decision.outcome == GateOutcome.ALLOW
    assert len(decision.surviving_claims) == 1
    assert len(decision.struck_claims) == 0


def test_gate_invariant_i2_fails_closed_when_entailment_unavailable():
    """Safety components never fail open (I2): unavailable verifier -> REFUSE."""
    claim = Claim(text="The agreement was signed in New Delhi.", passage_ids=["p1"], citations=[])
    struct = [StructuralResult(claim_index=0, is_valid=True, passage_ids=["p1"], status="ok")]
    entail = [make_mock_entailment(claim.text, is_entailed=True, status="unavailable")]

    decision = decide_gate([claim], struct, entail, as_of=TODAY)
    assert decision.outcome == GateOutcome.REFUSE
    assert "entailment" in decision.unavailable_components
    assert "safety_component_unavailable" in decision.reason


def test_gate_invariant_i2_fails_closed_when_structural_unavailable():
    """Safety components never fail open (I2): unavailable structural verifier -> REFUSE."""
    claim = Claim(text="Some legal claim.", passage_ids=["p1"])
    struct = [StructuralResult(claim_index=0, is_valid=False, status="unavailable")]
    entail = [make_mock_entailment(claim.text, is_entailed=True)]

    decision = decide_gate([claim], struct, entail, as_of=TODAY)
    assert decision.outcome == GateOutcome.REFUSE
    assert "structural" in decision.unavailable_components


def test_gate_refuses_when_all_claims_fail_entailment():
    claim = Claim(text="An unsupported assertion.", passage_ids=["p1"])
    struct = [StructuralResult(claim_index=0, is_valid=True, passage_ids=["p1"], status="ok")]
    entail = [make_mock_entailment(claim.text, is_entailed=False)]

    decision = decide_gate([claim], struct, entail, as_of=TODAY)
    assert decision.outcome == GateOutcome.REFUSE
    assert len(decision.struck_claims) == 1
    assert decision.struck_claims[0].component == "entailment"


def test_gate_refuses_when_all_claims_fail_structural():
    claim = Claim(text="Uncited claim.", passage_ids=[])
    struct = [StructuralResult(claim_index=0, is_valid=False, status="failed", reason="missing_passage_id")]
    entail = [make_mock_entailment(claim.text, is_entailed=True)]

    decision = decide_gate([claim], struct, entail, as_of=TODAY)
    assert decision.outcome == GateOutcome.REFUSE
    assert len(decision.struck_claims) == 1
    assert decision.struck_claims[0].component == "structural"


def test_gate_strikes_repealed_statutory_citation_as_of_today():
    """M4-M5 Integration: Section 420 IPC cited today must be struck by gate."""
    claim = Claim(text="The accused was charged under Section 420 IPC.", passage_ids=["p1"], citations=["Section 420 IPC"])
    struct = [StructuralResult(claim_index=0, is_valid=True, passage_ids=["p1"])]
    entail = [make_mock_entailment(claim.text, is_entailed=True)]

    decision = decide_gate([claim], struct, entail, as_of=TODAY)
    assert decision.outcome == GateOutcome.REFUSE
    assert len(decision.struck_claims) == 1
    struck = decision.struck_claims[0]
    assert struck.component == "registry"
    assert "repealed 2024-07-01" in struck.detail
    assert "BNS 318" in struck.detail


def test_gate_allows_current_bns_citation_as_of_today():
    """Section 318 BNS cited today is in force -> ALLOW."""
    claim = Claim(text="The offence falls under Section 318 BNS.", passage_ids=["p1"], citations=["Section 318 BNS"])
    struct = [StructuralResult(claim_index=0, is_valid=True, passage_ids=["p1"])]
    entail = [make_mock_entailment(claim.text, is_entailed=True)]

    decision = decide_gate([claim], struct, entail, as_of=TODAY)
    assert decision.outcome == GateOutcome.ALLOW
    assert len(decision.surviving_claims) == 1


def test_gate_partial_allow_when_one_claim_valid_and_one_struck():
    """Partial allow produces surviving claims and explicit gap notices."""
    c1 = Claim(text="Valid claim.", passage_ids=["p1"], citations=[])
    c2 = Claim(text="Invalid repealed claim citing Section 420 IPC.", passage_ids=["p1"], citations=["Section 420 IPC"])

    struct = [
        StructuralResult(claim_index=0, is_valid=True, passage_ids=["p1"]),
        StructuralResult(claim_index=1, is_valid=True, passage_ids=["p1"]),
    ]
    entail = [
        make_mock_entailment(c1.text, is_entailed=True),
        make_mock_entailment(c2.text, is_entailed=True),
    ]

    decision = decide_gate([c1, c2], struct, entail, as_of=TODAY)
    assert decision.outcome == GateOutcome.PARTIAL_ALLOW
    assert len(decision.surviving_claims) == 1
    assert decision.surviving_claims[0].text == "Valid claim."
    assert len(decision.struck_claims) == 1
    assert len(decision.gap_notices) > 0
    rendered = decision.render_answer()
    assert "Valid claim." in rendered
    assert "Notice:" in rendered


def test_gate_strikes_unresolvable_citation():
    """Unknown or fictitious citation -> struck."""
    claim = Claim(text="Citing Section 9999 IPC.", passage_ids=["p1"], citations=["Section 9999 IPC"])
    struct = [StructuralResult(claim_index=0, is_valid=True, passage_ids=["p1"])]
    entail = [make_mock_entailment(claim.text, is_entailed=True)]

    decision = decide_gate([claim], struct, entail, as_of=TODAY)
    assert decision.outcome == GateOutcome.REFUSE
    assert decision.struck_claims[0].component == "registry"
    assert "could not be resolved" in decision.struck_claims[0].detail
