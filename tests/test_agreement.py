"""Inter-annotator agreement tests (M6 Phase 4) -- RESEARCH.md §7.5.

Cohen's kappa and Krippendorff's alpha are hand-implemented here (see
eval/annotate/agreement.py for why), so they are checked against worked examples
with independently computable answers rather than against themselves.
"""
from __future__ import annotations

import pytest

from eval.annotate.agreement import (
    agreement,
    anchoring_report,
    cohen_kappa,
    krippendorff_alpha,
    percent_agreement,
)


# --------------------------------------------------------------------------- #
# Cohen's kappa
# --------------------------------------------------------------------------- #
def test_kappa_perfect_agreement_on_a_balanced_set():
    labels = ["a", "b", "a", "b"]
    assert cohen_kappa(labels, labels).value == 1.0


def test_kappa_total_disagreement_is_negative():
    assert cohen_kappa(["a", "b"], ["b", "a"]).value < 0


def test_kappa_hand_computed_case():
    # 20 items. Po = 16/20 = 0.8.
    # Rater A: 12 'a', 8 'b'. Rater B: 12 'a', 8 'b'.
    # Pe = 0.6*0.6 + 0.4*0.4 = 0.52. kappa = (0.8-0.52)/(1-0.52) = 0.5833
    a = ["a"] * 12 + ["b"] * 8
    b = ["a"] * 10 + ["b"] * 2 + ["a"] * 2 + ["b"] * 6
    r = cohen_kappa(a, b)
    assert r.value == pytest.approx(0.5833, abs=1e-3)
    assert r.n_items == 20 and r.n_raters == 2


def test_kappa_corrects_for_chance():
    """Agreement by coin-flip should land near zero, not near 0.5."""
    a = ["x", "y"] * 25
    b = ["x", "x", "y", "y"] * 12 + ["x", "y"]
    assert abs(cohen_kappa(a, b).value) < 0.25


def test_kappa_is_undefined_when_both_raters_use_one_label():
    """Chance agreement is already total, so kappa is 0/0 -- not 1.0."""
    r = cohen_kappa(["a"] * 5, ["a"] * 5)
    assert r.value is None
    assert r.status == "unavailable"
    assert "single_label" in r.reason


def test_kappa_rejects_misaligned_or_empty_input():
    assert cohen_kappa([], []).status == "unavailable"
    assert cohen_kappa(["a"], ["a", "b"]).status == "unavailable"


# --------------------------------------------------------------------------- #
# Krippendorff's alpha
# --------------------------------------------------------------------------- #
def test_alpha_perfect_agreement_across_three_raters():
    data = {"i1": {"r1": "a", "r2": "a", "r3": "a"},
            "i2": {"r1": "b", "r2": "b", "r3": "b"},
            "i3": {"r1": "a", "r2": "a", "r3": "a"}}
    assert krippendorff_alpha(data).value == 1.0


def test_alpha_drops_below_one_on_disagreement():
    data = {"i1": {"r1": "a", "r2": "a", "r3": "b"},
            "i2": {"r1": "b", "r2": "b", "r3": "b"},
            "i3": {"r1": "a", "r2": "b", "r3": "a"}}
    value = krippendorff_alpha(data).value
    assert 0.0 < value < 1.0


def test_alpha_is_near_zero_for_chance_level_labelling():
    data = {f"i{i}": {"r1": "a" if i % 2 else "b",
                      "r2": "a" if i % 3 else "b"} for i in range(24)}
    assert abs(krippendorff_alpha(data).value) < 0.4


def test_alpha_handles_incomplete_designs():
    """Only a fraction of each set is duplicated (RESEARCH.md §7.1.3), so alpha
    must cope with items rated by different numbers of raters."""
    data = {"i1": {"r1": "a", "r2": "a", "r3": "a"},
            "i2": {"r1": "b", "r2": "b"},
            "i3": {"r1": "a"}}                     # singly rated -> excluded
    r = krippendorff_alpha(data)
    assert r.n_items == 2
    assert r.value is not None


def test_alpha_is_unavailable_when_nothing_is_duplicated():
    r = krippendorff_alpha({"i1": {"r1": "a"}, "i2": {"r1": "b"}})
    assert r.value is None
    assert "two_or_more" in r.reason


def test_alpha_is_undefined_when_only_one_label_exists():
    """No disagreement was possible, so alpha carries no information. Reporting
    1.0 would claim perfect agreement nothing could have detected."""
    data = {"i1": {"r1": "a", "r2": "a"}, "i2": {"r1": "a", "r2": "a"}}
    r = krippendorff_alpha(data)
    assert r.value is None
    assert "undefined" in r.reason


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def test_agreement_uses_kappa_for_two_raters():
    data = {"i1": {"r1": "a", "r2": "b"}, "i2": {"r1": "b", "r2": "b"},
            "i3": {"r1": "a", "r2": "a"}}
    assert agreement(data).statistic == "cohen_kappa"


def test_agreement_uses_alpha_for_three_or_more_raters():
    data = {"i1": {"r1": "a", "r2": "b", "r3": "a"},
            "i2": {"r1": "b", "r2": "b", "r3": "a"}}
    assert agreement(data).statistic == "krippendorff_alpha"


def test_agreement_is_unavailable_with_one_rater():
    r = agreement({"i1": {"r1": "a"}})
    assert r.value is None and "fewer_than_two" in r.reason


def test_agreement_never_raises_on_odd_input():
    """The app calls this after every submission; it must not crash a session."""
    for data in ({}, {"i1": {}}, {"i1": {"r1": "a"}, "i2": {"r2": "b"}}):
        assert agreement(data).status in ("ok", "unavailable")


def test_agreement_handles_two_raters_with_no_common_item():
    r = agreement({"i1": {"r1": "a"}, "i2": {"r2": "b"}})
    assert r.value is None and "no_commonly_rated" in r.reason


def test_percent_agreement_is_reported_but_flatters_skew():
    """95% raw agreement on a 95/5 label split is chance -- kappa says so."""
    data = {f"i{i}": {"r1": "a", "r2": "a"} for i in range(19)}
    data["i19"] = {"r1": "a", "r2": "b"}
    assert percent_agreement(data).value == 0.95
    assert krippendorff_alpha(data).value < 0.5


def test_percent_agreement_unavailable_without_duplication():
    assert percent_agreement({"i1": {"r1": "a"}}).status == "unavailable"


# --------------------------------------------------------------------------- #
# Anchoring (RESEARCH.md §7.1.1)
# --------------------------------------------------------------------------- #
def test_anchoring_detects_a_gap_between_assisted_and_cold_labels():
    assisted = [("a", "a")] * 10                              # always accepts
    cold = [("a", "a"), ("b", "a"), ("b", "a"), ("a", "a")]   # 50% agreement
    r = anchoring_report(assisted, cold)
    assert r.assisted_agreement_with_model == 1.0
    assert r.cold_agreement_with_model == 0.5
    assert r.gap == 0.5


def test_anchoring_reports_no_gap_when_there_is_none():
    pairs = [("a", "a"), ("b", "a")]
    assert anchoring_report(pairs, pairs).gap == 0.0


def test_anchoring_does_not_decide_whether_the_gap_is_acceptable():
    """A threshold here would turn a reportable finding into a silent pass/fail."""
    r = anchoring_report([("a", "a")] * 5, [("b", "a")] * 5)
    assert not hasattr(r, "passed")
    assert r.gap == 1.0


def test_anchoring_needs_both_subsets():
    r = anchoring_report([("a", "a")], [])
    assert r.status == "unavailable"
    assert r.gap is None
