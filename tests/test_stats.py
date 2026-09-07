"""Statistical protocol tests (M6 Phase 11) -- RESEARCH.md §6.

These are the functions that turn metric numbers into claims a reviewer is
asked to believe, so they get worked examples and known-answer checks, not
smoke tests.
"""
from __future__ import annotations

import pytest

from eval import config
from eval.stats import (
    bonferroni_alpha,
    bonferroni_correct,
    expected_calibration_error,
    mean_std,
    paired_bootstrap,
    proportion_ci,
    reliability_diagram_data,
)


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_detects_a_real_difference():
    a = [0.9] * 30
    b = [0.5] * 30
    r = paired_bootstrap(a, b, n_resamples=2000)
    assert r.mean_difference == pytest.approx(0.4)
    assert r.ci_lower > 0
    assert r.significant


def test_bootstrap_ci_contains_zero_for_identical_systems():
    scores = [0.1, 0.5, 0.9, 0.3, 0.7] * 6
    r = paired_bootstrap(scores, scores, n_resamples=2000)
    assert r.mean_difference == 0.0
    assert r.ci_lower <= 0 <= r.ci_upper
    assert not r.significant


def test_bootstrap_is_not_significant_for_a_noisy_tiny_difference():
    a = [0.50, 0.80, 0.20, 0.95, 0.05, 0.60, 0.40, 0.70]
    b = [0.52, 0.10, 0.90, 0.05, 0.95, 0.20, 0.85, 0.15]
    r = paired_bootstrap(a, b, n_resamples=3000)
    assert not r.significant
    assert r.ci_lower < 0 < r.ci_upper


def test_bootstrap_uses_the_pairing():
    """Paired resampling must draw item indices, applying them to BOTH systems.
    Here every item differs by exactly +0.1, so the paired CI is a point at 0.1;
    an unpaired procedure would produce a visibly wider interval."""
    a = [0.1, 0.4, 0.9, 0.2, 0.6, 0.8]
    b = [x - 0.1 for x in a]
    r = paired_bootstrap(a, b, n_resamples=2000)
    assert r.ci_lower == pytest.approx(0.1, abs=1e-6)
    assert r.ci_upper == pytest.approx(0.1, abs=1e-6)


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    a, b = [0.1, 0.9, 0.4, 0.6], [0.2, 0.3, 0.8, 0.1]
    first = paired_bootstrap(a, b, n_resamples=1000, seed=42)
    second = paired_bootstrap(a, b, n_resamples=1000, seed=42)
    assert first.to_dict() == second.to_dict()


def test_bootstrap_seed_changes_the_interval():
    a, b = [0.1, 0.9, 0.4, 0.6, 0.2], [0.2, 0.3, 0.8, 0.1, 0.7]
    assert (paired_bootstrap(a, b, n_resamples=500, seed=42).ci_lower
            != paired_bootstrap(a, b, n_resamples=500, seed=1337).ci_lower)


def test_bootstrap_reports_effect_size():
    """RESEARCH.md §6: report effect size, not just significance."""
    r = paired_bootstrap([0.9, 0.8, 0.85, 0.95], [0.5, 0.4, 0.45, 0.55],
                         n_resamples=500)
    assert r.effect_size is not None and r.effect_size > 1.0


def test_bootstrap_effect_size_is_none_when_differences_have_no_spread():
    r = paired_bootstrap([0.5, 0.5, 0.5], [0.4, 0.4, 0.4], n_resamples=200)
    assert r.effect_size is None


def test_bootstrap_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="paired"):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_bootstrap_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        paired_bootstrap([], [])


def test_bootstrap_default_resamples_matches_the_protocol():
    assert config.BOOTSTRAP_RESAMPLES == 10_000


def test_bootstrap_summary_is_human_readable():
    s = paired_bootstrap([0.9, 0.8], [0.5, 0.4], n_resamples=200).summary()
    assert "95% CI" in s and "diff=" in s


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_ece_is_zero_for_a_perfectly_calibrated_system():
    # Bin [0.9,1.0): confidence 1.0, all correct. Bin [0.0,0.1): 0.0, none correct.
    confidences = [1.0] * 10 + [0.0] * 10
    correctness = [True] * 10 + [False] * 10
    assert expected_calibration_error(confidences, correctness) == 0.0


def test_ece_is_one_for_maximal_miscalibration():
    confidences = [1.0] * 10
    correctness = [False] * 10
    assert expected_calibration_error(confidences, correctness) == 1.0


def test_ece_hand_computed_case():
    # 4 items at confidence 1.0, half correct -> |1.0 - 0.5| = 0.5, one bin.
    assert expected_calibration_error([1.0, 1.0, 1.0, 1.0],
                                      [True, True, False, False]) == 0.5


def test_ece_weights_bins_by_count():
    # Bin A: 9 items conf 1.0 all correct (gap 0). Bin B: 1 item conf 0.05
    # correct (gap 0.95). Weighted ECE = 0.1 * 0.95 = 0.095.
    confidences = [1.0] * 9 + [0.05]
    correctness = [True] * 9 + [True]
    assert expected_calibration_error(confidences, correctness) == pytest.approx(
        0.095, abs=1e-3)


def test_ece_of_nothing_is_none_not_zero():
    """A perfectly calibrated system and an unmeasured one must not print the
    same number (I2)."""
    assert expected_calibration_error([], []) is None


def test_ece_ignores_none_confidences():
    """I2: an unavailable verifier score is not a confidence of 0.0."""
    assert expected_calibration_error([None, 1.0, 1.0], [True, True, True]) == 0.0


def test_reliability_bins_count_every_item():
    confidences = [i / 20 for i in range(20)]
    correctness = [i % 2 == 0 for i in range(20)]
    bins = reliability_diagram_data(confidences, correctness)
    assert sum(b["count"] for b in bins) == 20


def test_reliability_omits_empty_bins():
    bins = reliability_diagram_data([0.95, 0.97], [True, False])
    assert len(bins) == 1
    assert bins[0]["count"] == 2
    assert bins[0]["accuracy"] == 0.5


def test_reliability_bins_are_ordered_and_bounded():
    bins = reliability_diagram_data([0.05, 0.55, 0.95], [True, False, True])
    assert [b["bin_lower"] for b in bins] == sorted(b["bin_lower"] for b in bins)
    for b in bins:
        assert 0.0 <= b["bin_lower"] < b["bin_upper"] <= 1.0


def test_reliability_clamps_out_of_range_confidences():
    bins = reliability_diagram_data([1.5, -0.5], [True, False])
    assert sum(b["count"] for b in bins) == 2


def test_reliability_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="align"):
        reliability_diagram_data([0.5], [True, False])


# --------------------------------------------------------------------------- #
# Multiple comparisons
# --------------------------------------------------------------------------- #
def test_bonferroni_is_stricter_than_uncorrected():
    p_values = [0.01, 0.02, 0.04, 0.30]
    uncorrected = [p <= 0.05 for p in p_values]
    corrected = bonferroni_correct(p_values)
    assert sum(corrected) < sum(uncorrected)
    assert corrected == [True, False, False, False]  # threshold 0.05/4 = 0.0125


def test_bonferroni_single_comparison_is_uncorrected():
    assert bonferroni_correct([0.04]) == [True]


def test_bonferroni_empty():
    assert bonferroni_correct([]) == []


def test_bonferroni_alpha_divides_by_the_comparison_count():
    assert bonferroni_alpha(4) == 0.0125
    assert bonferroni_alpha(0) == config.ALPHA


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def test_mean_std_over_the_five_pre_registered_seeds():
    r = mean_std([0.70, 0.72, 0.68, 0.71, 0.69])
    assert r.n == 5
    assert r.mean == pytest.approx(0.70, abs=1e-4)
    assert r.std is not None and r.std > 0
    assert "+/-" in str(r)


def test_single_run_has_no_std_rather_than_zero():
    """One run has no spread; printing 0.0 would claim it does."""
    r = mean_std([0.7])
    assert r.std is None
    assert "no std" in str(r)


def test_mean_std_of_nothing_is_unavailable():
    r = mean_std([])
    assert r.mean is None and str(r) == "unavailable"


def test_mean_std_drops_none_values():
    assert mean_std([0.5, None, 0.7]).n == 2


def test_five_seeds_are_pre_registered():
    assert config.SEEDS == (42, 1337, 2024, 31337, 8191)


def test_wilson_interval_stays_inside_zero_one_at_the_boundary():
    """Why Wilson and not normal-approximation: several rates here sit at 0 or 1."""
    low, high = proportion_ci(0, 20)
    assert low == 0.0 and 0.0 < high < 1.0
    low, high = proportion_ci(20, 20)
    assert high == 1.0 and 0.0 < low < 1.0


def test_wilson_interval_brackets_the_point_estimate():
    low, high = proportion_ci(15, 30)
    assert low < 0.5 < high


def test_wilson_interval_of_nothing_is_none():
    assert proportion_ci(0, 0) is None
