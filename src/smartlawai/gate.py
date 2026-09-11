"""Deterministic gate decision engine.

Implements:
- Invariant I2: Safety components never fail open (any unavailable -> REFUSE).
- Invariant I3: Refusal decision is deterministic Python application code, never an LLM.
- Invariant I12: Classifiers and deterministic registry lookups only, no generative judge.

Gate rules (PLAN.md M5.4):
    structural AND entailment AND registry -> ALLOW
    any component unavailable             -> REFUSE
    all claims struck                     -> REFUSE
    some struck                           -> PARTIAL_ALLOW + explicit gap notice
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from smartlawai.protocols import Claim
from smartlawai.registry.types import (
    STATUS_IN_FORCE,
    RegistryResult,
)
from smartlawai.verify.entailment import ClaimEntailmentResult
from smartlawai.verify.registry_check import check_citation
from smartlawai.verify.structural import StructuralResult


class GateOutcome(str, Enum):
    ALLOW = "ALLOW"
    PARTIAL_ALLOW = "PARTIAL_ALLOW"
    REFUSE = "REFUSE"


@dataclass
class StruckClaim:
    """Record of a claim struck by the gate with explanation."""

    claim: Claim
    reason: str
    component: str  # "structural" | "entailment" | "registry"
    detail: str = ""


@dataclass
class GateDecision:
    """Deterministic refusal/allowance decision record."""

    outcome: GateOutcome
    surviving_claims: list[Claim] = field(default_factory=list)
    struck_claims: list[StruckClaim] = field(default_factory=list)
    unavailable_components: list[str] = field(default_factory=list)
    gap_notices: list[str] = field(default_factory=list)
    reason: str = "ok"

    def render_answer(self, fallback_message: str = "I cannot verify an answer from the available document(s).") -> str:
        """Render final verified answer text with explicit gap notices for struck claims."""
        if self.outcome == GateOutcome.REFUSE:
            return fallback_message

        parts = [c.text for c in self.surviving_claims]
        if self.gap_notices:
            parts.append("\n[Notice: Certain assertions were omitted because: " + "; ".join(self.gap_notices) + "]")
        return " ".join(parts).strip()


def decide_gate(
    claims: list[Claim],
    structural_results: list[StructuralResult] | None,
    entailment_results: list[ClaimEntailmentResult] | None,
    as_of: date | None = None,
    registry_checker=None,
) -> GateDecision:
    """Evaluate deterministic gate conjunction on candidate claims.

    Guarantees:
    - Never raises (I2).
    - Refuses if any verifier component is unavailable (I2).
    - Never uses an LLM to decide (I3).
    """
    if not claims:
        return GateDecision(
            outcome=GateOutcome.REFUSE,
            reason="no_claims_to_verify",
        )

    as_of_date = as_of or date.today()  # noqa: DTZ011
    check_fn = registry_checker or check_citation
    unavailable: list[str] = []

    # Check structural component availability
    if structural_results is None or any(sr.status == "unavailable" for sr in structural_results):
        unavailable.append("structural")

    # Check entailment component availability
    if entailment_results is None or any(er.status == "unavailable" for er in entailment_results):
        unavailable.append("entailment")

    # INVARIANT I2: Any unavailable component -> Gate MUST REFUSE
    if unavailable:
        return GateDecision(
            outcome=GateOutcome.REFUSE,
            unavailable_components=unavailable,
            reason=f"safety_component_unavailable: {', '.join(unavailable)}",
        )

    surviving: list[Claim] = []
    struck: list[StruckClaim] = []
    gap_notices: list[str] = []

    from smartlawai.registry.resolver import resolve_citation

    for idx, claim in enumerate(claims):
        # 1. Structural check
        sr = structural_results[idx]
        if not sr.is_valid:
            sc = StruckClaim(
                claim=claim,
                reason="failed_structural_grounding",
                component="structural",
                detail=sr.reason,
            )
            struck.append(sc)
            gap_notices.append(f"Claim lacks valid document citation ({sr.reason})")
            continue

        # 2. Entailment check
        er = entailment_results[idx]
        if not er.is_entailed:
            sc = StruckClaim(
                claim=claim,
                reason="not_entailed_by_passage",
                component="entailment",
                detail=er.reason,
            )
            struck.append(sc)
            gap_notices.append("Claim assertion is not factually entailed by the cited passage")
            continue

        # 3. Registry authority check for cited provisions
        citation_struck = False
        if claim.citations:
            for raw_cit in claim.citations:
                resolved = resolve_citation(raw_cit)
                if resolved is None:
                    # Unresolved / invalid citation -> strike claim (I2)
                    sc = StruckClaim(
                        claim=claim,
                        reason="unresolved_statutory_citation",
                        component="registry",
                        detail=f"Citation '{raw_cit}' could not be resolved in authority registry",
                    )
                    struck.append(sc)
                    gap_notices.append(f"Cited authority '{raw_cit}' was not found in statutory registry")
                    citation_struck = True
                    break

                reg_res: RegistryResult = check_fn(resolved, as_of=as_of_date)
                if reg_res.status != STATUS_IN_FORCE:
                    # Not in force (superseded, repealed, or not_found)
                    sc = StruckClaim(
                        claim=claim,
                        reason=f"statute_not_in_force_{reg_res.status}",
                        component="registry",
                        detail=reg_res.note or f"Authority not in force on {as_of_date}",
                    )
                    struck.append(sc)
                    gap_notices.append(
                        f"Cited provision '{raw_cit}' is not in force as of {as_of_date}"
                        + (f" ({reg_res.note})" if reg_res.note else "")
                    )
                    citation_struck = True
                    break

        if not citation_struck:
            surviving.append(claim)

    # Conjunction evaluation
    if not surviving:
        return GateDecision(
            outcome=GateOutcome.REFUSE,
            struck_claims=struck,
            gap_notices=gap_notices,
            reason="all_claims_struck_by_verification",
        )
    elif len(struck) > 0:
        return GateDecision(
            outcome=GateOutcome.PARTIAL_ALLOW,
            surviving_claims=surviving,
            struck_claims=struck,
            gap_notices=gap_notices,
            reason="partial_claims_verified",
        )
    else:
        return GateDecision(
            outcome=GateOutcome.ALLOW,
            surviving_claims=surviving,
            struck_claims=[],
            gap_notices=[],
            reason="all_claims_verified",
        )
