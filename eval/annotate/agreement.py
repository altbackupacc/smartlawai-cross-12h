"""Inter-annotator agreement: Cohen's kappa, Krippendorff's alpha, anchoring.

RESEARCH.md §7.5: "Report agreement for every study. Cohen's kappa (2 raters) /
Krippendorff's alpha (3+)." Used live by the annotation app as items come in
(PLAN.md M6) and again standalone when a study is written up.

Hand-implemented deliberately, and this is the one place in M6 where that is
correct: there is no maintained, lightweight Krippendorff implementation in this
project's dependency set, and the alternative -- adding a dependency for eight
lines of arithmetic -- conflicts with the single-purpose extras discipline
`pyproject.toml` documents. Both are textbook formulae with worked-example tests
in tests/test_agreement.py, which is what "never hand-rolled" is actually
protecting against (RESEARCH.md §5 names ROUGE and BERTScore, where subtle
tokenisation choices make independent implementations disagree).

Returns `None`, never a number, when agreement is undefined -- a single rater,
or one label used by everyone. Krippendorff's alpha is genuinely undefined when
observed disagreement is zero AND expected disagreement is zero; reporting 1.0
there would claim perfect agreement that no disagreement could ever have
detected. RESEARCH.md §7.5 expects some of these numbers to be ugly; fabricating
a clean one is worse than reporting None.

Imports nothing from `smartlawai` (OPS.md §8).
"""
from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgreementResult:
    statistic: str          # "cohen_kappa" | "krippendorff_alpha" | "percent"
    value: float | None
    n_items: int
    n_raters: int
    status: str = "ok"      # "ok" | "unavailable"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"statistic": self.statistic, "value": self.value,
                "n_items": self.n_items, "n_raters": self.n_raters,
                "status": self.status, "reason": self.reason}

    @classmethod
    def unavailable(cls, statistic: str, reason: str, n_items: int = 0,
                    n_raters: int = 0) -> AgreementResult:
        return cls(statistic=statistic, value=None, n_items=n_items,
                   n_raters=n_raters, status="unavailable", reason=reason)


# --------------------------------------------------------------------------- #
# Pairwise: Cohen's kappa
# --------------------------------------------------------------------------- #
def cohen_kappa(labels_a: list[Hashable], labels_b: list[Hashable]) -> AgreementResult:
    """Cohen's kappa for two raters over the same items, in the same order.

    kappa = (Po - Pe) / (1 - Pe), where Po is observed agreement and Pe is the
    agreement expected from the two raters' marginal label distributions.
    """
    n = len(labels_a)
    if n == 0 or n != len(labels_b):
        return AgreementResult.unavailable("cohen_kappa", "no_or_misaligned_items",
                                           n_items=n, n_raters=2)

    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    count_a, count_b = Counter(labels_a), Counter(labels_b)
    pe = sum((count_a[label] / n) * (count_b[label] / n)
             for label in set(count_a) | set(count_b))

    if pe == 1.0:
        # Both raters used exactly one label for everything: chance agreement is
        # already perfect, so kappa is 0/0. Undefined, not 1.0.
        return AgreementResult.unavailable(
            "cohen_kappa", "chance_agreement_is_total_single_label_used",
            n_items=n, n_raters=2)
    return AgreementResult("cohen_kappa", round((po - pe) / (1 - pe), 4), n, 2)


# --------------------------------------------------------------------------- #
# Multi-rater: Krippendorff's alpha (nominal)
# --------------------------------------------------------------------------- #
def krippendorff_alpha(annotations: dict[Hashable, dict[Hashable, Hashable]],
                       ) -> AgreementResult:
    """Krippendorff's alpha with a nominal difference function.

    annotations: {item_id: {rater_id: label}}. Missing ratings are simply absent
    -- alpha handles incomplete designs natively, which matters here because
    only a fraction of each set is duplicated (RESEARCH.md §7.1.3).

    alpha = 1 - Do/De, computed via the coincidence matrix so that items rated
    by different numbers of raters are weighted correctly.
    """
    units = {i: r for i, r in annotations.items() if len(r) >= 2}
    all_raters = {rater for ratings in annotations.values() for rater in ratings}
    n_items, n_raters = len(units), len(all_raters)

    if not units:
        return AgreementResult.unavailable(
            "krippendorff_alpha", "no_item_has_two_or_more_ratings",
            n_items=len(annotations), n_raters=n_raters)

    # Coincidence matrix: each ordered pair within a unit contributes 1/(m_u - 1).
    coincidences: Counter = Counter()
    for ratings in units.values():
        values = list(ratings.values())
        m_u = len(values)
        weight = 1.0 / (m_u - 1)
        for v1, v2 in itertools.permutations(values, 2):
            coincidences[(v1, v2)] += weight

    n_total = sum(coincidences.values())
    if n_total == 0:
        return AgreementResult.unavailable("krippendorff_alpha", "empty_coincidence_matrix",
                                           n_items=n_items, n_raters=n_raters)

    marginals: Counter = Counter()
    for (v1, _), count in coincidences.items():
        marginals[v1] += count

    # Nominal metric: disagreement is 1 for any unequal pair, 0 otherwise.
    observed = sum(c for (v1, v2), c in coincidences.items() if v1 != v2)
    expected = sum(marginals[v1] * marginals[v2]
                   for v1, v2 in itertools.permutations(marginals, 2))

    if expected == 0:
        # Only one distinct label exists across the whole set. No disagreement
        # was possible, so alpha carries no information -- report that, not 1.0.
        return AgreementResult.unavailable(
            "krippendorff_alpha", "single_label_across_all_units_alpha_undefined",
            n_items=n_items, n_raters=n_raters)

    do = observed / n_total
    de = expected / (n_total * (n_total - 1))
    return AgreementResult("krippendorff_alpha", round(1.0 - do / de, 4),
                           n_items, n_raters)


# --------------------------------------------------------------------------- #
# Dispatch + live use
# --------------------------------------------------------------------------- #
def percent_agreement(annotations: dict[Hashable, dict[Hashable, Hashable]],
                      ) -> AgreementResult:
    """Raw pairwise agreement. Reported alongside kappa/alpha, never instead of
    them -- percent agreement flatters skewed label distributions."""
    units = {i: r for i, r in annotations.items() if len(r) >= 2}
    if not units:
        return AgreementResult.unavailable("percent", "no_duplicated_items",
                                           n_items=len(annotations))
    agree = total = 0
    for ratings in units.values():
        for v1, v2 in itertools.combinations(ratings.values(), 2):
            total += 1
            agree += int(v1 == v2)
    raters = {r for ratings in annotations.values() for r in ratings}
    if total == 0:
        return AgreementResult.unavailable("percent", "no_rater_pairs",
                                           n_items=len(units), n_raters=len(raters))
    return AgreementResult("percent", round(agree / total, 4), len(units), len(raters))


def agreement(annotations: dict[Hashable, dict[Hashable, Hashable]]) -> AgreementResult:
    """Pick the right statistic: Cohen's kappa for exactly 2 raters, else alpha.

    This is what the app calls after every submission (PLAN.md M6: agreement
    "shown to whoever's running the session as items come in, not just at the
    end"), so it must be cheap and must never raise.
    """
    raters = sorted({r for ratings in annotations.values() for r in ratings},
                    key=str)
    if len(raters) < 2:
        return AgreementResult.unavailable("cohen_kappa", "fewer_than_two_raters",
                                           n_items=len(annotations),
                                           n_raters=len(raters))
    if len(raters) == 2:
        a, b = raters
        paired = [(v[a], v[b]) for v in annotations.values() if a in v and b in v]
        if not paired:
            return AgreementResult.unavailable("cohen_kappa", "no_commonly_rated_items",
                                               n_items=len(annotations), n_raters=2)
        return cohen_kappa([p[0] for p in paired], [p[1] for p in paired])
    return krippendorff_alpha(annotations)


# --------------------------------------------------------------------------- #
# Anchoring control (RESEARCH.md §7.1.1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnchoringReport:
    """Do model-assisted labels agree with the model more than cold labels do?

    RESEARCH.md §7.1.1: model-assisted pre-labelling buys 2-3x throughput, but
    "anchoring must be controlled: have one rater label a 10% subset cold and
    compare against the assisted labels. If assisted labels agree with the model
    far more than the cold subset does, you have anchoring -- report it and
    discount accordingly."

    This computes the comparison. It deliberately does NOT decide whether the
    gap is acceptable: that is a judgment for whoever writes up the study, and
    a threshold here would turn a reportable finding into a silent pass/fail.
    """

    assisted_agreement_with_model: float | None
    cold_agreement_with_model: float | None
    gap: float | None
    n_assisted: int
    n_cold: int
    status: str = "ok"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"assisted_agreement_with_model": self.assisted_agreement_with_model,
                "cold_agreement_with_model": self.cold_agreement_with_model,
                "gap": self.gap, "n_assisted": self.n_assisted, "n_cold": self.n_cold,
                "status": self.status, "reason": self.reason}


def anchoring_report(assisted: list[tuple[Hashable, Hashable]],
                     cold: list[tuple[Hashable, Hashable]]) -> AnchoringReport:
    """assisted / cold: lists of (human_label, model_label) for each subset."""
    if not assisted or not cold:
        return AnchoringReport(None, None, None, len(assisted), len(cold),
                               status="unavailable",
                               reason="need_both_an_assisted_and_a_cold_subset")
    a_rate = sum(h == m for h, m in assisted) / len(assisted)
    c_rate = sum(h == m for h, m in cold) / len(cold)
    return AnchoringReport(round(a_rate, 4), round(c_rate, 4),
                           round(a_rate - c_rate, 4), len(assisted), len(cold))
