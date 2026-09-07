"""RESEARCH.md §6's statistical protocol, as code.

Four things the paper needs and cannot get from a metric alone:

  paired_bootstrap        -- system comparison over the SAME test items, 10k
                             resamples. Explicitly not a t-test over 5 seeds,
                             which RESEARCH.md §6 calls underpowered for this.
  expected_calibration_error / reliability_diagram_data
                          -- mandatory because the product shows a faithfulness
                             score to non-experts making consequential decisions
                             (RESEARCH.md §5). Does 0.8 mean 80% correct?
  bonferroni_correct      -- applied automatically when more than two systems are
                             compared on one metric, so it is not a manual step
                             someone forgets in run_eval.py.
  mean_std                -- "mean +/- std always" (§6), including across the
                             five pre-registered seeds.

Seeded everywhere: the same inputs give the same CI on any machine, because a
number in the paper that moves between runs cannot be checked by a reviewer.

Uses numpy (already a core dependency) rather than scipy for the resampling
loop: scipy is in the [eval] extra for callers who want it, but nothing here
needs a scipy-only routine, and a hard scipy import would make the whole stats
module unavailable to anyone who installed only the base package.

Imports nothing from `smartlawai` (OPS.md §8).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval import config


@dataclass
class BootstrapResult:
    """Paired-bootstrap comparison of two systems on the same items."""

    mean_a: float
    mean_b: float
    mean_difference: float          # a - b
    ci_lower: float
    ci_upper: float
    confidence: float
    n_items: int
    n_resamples: int
    seed: int
    significant: bool               # CI excludes zero
    effect_size: float | None       # Cohen's d over the paired differences

    def to_dict(self) -> dict[str, Any]:
        return {"mean_a": self.mean_a, "mean_b": self.mean_b,
                "mean_difference": self.mean_difference,
                "ci_lower": self.ci_lower, "ci_upper": self.ci_upper,
                "confidence": self.confidence, "n_items": self.n_items,
                "n_resamples": self.n_resamples, "seed": self.seed,
                "significant": self.significant, "effect_size": self.effect_size}

    def summary(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return (f"diff={self.mean_difference:+.4f} "
                f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}] "
                f"({self.confidence:.0%} CI, n={self.n_items}, {verdict}, "
                f"d={self.effect_size if self.effect_size is None else round(self.effect_size, 3)})")


def paired_bootstrap(scores_a: Sequence[float], scores_b: Sequence[float],
                     n_resamples: int = config.BOOTSTRAP_RESAMPLES,
                     seed: int = config.DEFAULT_SEED,
                     confidence: float = config.CONFIDENCE_LEVEL) -> BootstrapResult:
    """Paired bootstrap over test items for comparing two systems on the same items.

    RESEARCH.md §6 -- NOT a t-test over 5 seeds, which is underpowered for this
    kind of comparison. Returns the mean difference and a 95% CI; a CI that
    excludes zero is the significance claim.

    Pairing is the point: `scores_a[i]` and `scores_b[i]` must be the same test
    item, so the resample draws item INDICES and applies them to both systems.
    Resampling the two independently would throw away the pairing and inflate
    the interval.

    Also reports Cohen's d over the paired differences, because RESEARCH.md §6
    requires effect size alongside significance ("a significant 0.3 ROUGE point
    is noise").
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"paired bootstrap needs paired inputs: "
                         f"{len(scores_a)} vs {len(scores_b)} scores")
    if not scores_a:
        raise ValueError("paired bootstrap needs at least one item")

    a = np.asarray(scores_a, dtype="float64")
    b = np.asarray(scores_b, dtype="float64")
    diffs = a - b
    n = len(a)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = diffs[idx].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(resampled, [tail, 1.0 - tail])

    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    effect = round(float(diffs.mean()) / sd, 4) if sd > 0 else None

    return BootstrapResult(
        mean_a=round(float(a.mean()), 4), mean_b=round(float(b.mean()), 4),
        mean_difference=round(float(diffs.mean()), 4),
        ci_lower=round(float(lower), 4), ci_upper=round(float(upper), 4),
        confidence=confidence, n_items=n, n_resamples=n_resamples, seed=seed,
        significant=bool(lower > 0 or upper < 0), effect_size=effect)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def expected_calibration_error(confidences: Sequence[float],
                               correctness: Sequence[bool],
                               n_bins: int = config.ECE_BINS) -> float | None:
    """ECE: bin predictions by confidence, compare each bin's mean confidence to
    its actual accuracy.

    RESEARCH.md §5: mandatory because the system displays a faithfulness score to
    non-expert users making real decisions.

    Returns None -- not 0.0 -- for empty input. A perfectly calibrated system and
    an unmeasured one both cannot be 0.0 in the same report (I2).
    """
    bins = reliability_diagram_data(confidences, correctness, n_bins)
    if not bins:
        return None
    total = sum(b["count"] for b in bins)
    if total == 0:
        return None
    return round(sum(b["count"] / total * abs(b["mean_confidence"] - b["accuracy"])
                     for b in bins), 4)


def reliability_diagram_data(confidences: Sequence[float],
                             correctness: Sequence[bool],
                             n_bins: int = config.ECE_BINS) -> list[dict]:
    """Per-bin (mean_confidence, accuracy, count) -- what a reliability diagram
    plots. Returns data, not a rendered figure; plotting is the report's job.

    Empty bins are omitted rather than emitted with accuracy 0.0, which would
    draw a misleading point at the bottom of the diagram.
    """
    if len(confidences) != len(correctness):
        raise ValueError(f"confidences and correctness must align: "
                         f"{len(confidences)} vs {len(correctness)}")
    if not confidences or n_bins <= 0:
        return []

    edges = [i / n_bins for i in range(n_bins + 1)]
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, correctness):
        if conf is None:
            continue  # I2: an unavailable score is not a confidence of 0.0
        c = min(max(float(conf), 0.0), 1.0)
        index = min(int(c * n_bins), n_bins - 1)
        buckets[index].append((c, bool(correct)))

    out: list[dict] = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        confs = [c for c, _ in bucket]
        hits = [correct for _, correct in bucket]
        out.append({
            "bin_lower": round(edges[i], 4),
            "bin_upper": round(edges[i + 1], 4),
            "mean_confidence": round(sum(confs) / len(confs), 4),
            "accuracy": round(sum(hits) / len(hits), 4),
            "count": len(bucket),
        })
    return out


# --------------------------------------------------------------------------- #
# Multiple comparisons
# --------------------------------------------------------------------------- #
def bonferroni_correct(p_values: Sequence[float],
                       alpha: float = config.ALPHA) -> list[bool]:
    """Bonferroni: reject H0_i iff p_i <= alpha / m.

    RESEARCH.md §6 requires this "where multiple systems are compared on one
    metric". Exposed as a function so run_eval.py applies it automatically
    rather than leaving it as a step someone has to remember.
    """
    m = len(p_values)
    if m == 0:
        return []
    threshold = alpha / m
    return [p <= threshold for p in p_values]


def bonferroni_alpha(n_comparisons: int, alpha: float = config.ALPHA) -> float:
    """The corrected per-comparison alpha, for reporting alongside results."""
    return alpha / n_comparisons if n_comparisons > 0 else alpha


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
@dataclass
class MeanStd:
    mean: float | None
    std: float | None
    n: int
    values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "n": self.n,
                "values": self.values}

    def __str__(self) -> str:
        if self.mean is None:
            return "unavailable"
        if self.std is None:
            return f"{self.mean:.4f} (n=1, no std)"
        return f"{self.mean:.4f} +/- {self.std:.4f}"


def mean_std(values: Sequence[float]) -> MeanStd:
    """"Mean +/- std always" (RESEARCH.md §6), typically over the five
    pre-registered seeds. A single value reports std=None rather than 0.0 --
    one run has no spread, and printing 0.0 would claim it does.
    """
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return MeanStd(None, None, 0, [])
    if len(vals) == 1:
        return MeanStd(round(vals[0], 4), None, 1, vals)
    arr = np.asarray(vals, dtype="float64")
    return MeanStd(round(float(arr.mean()), 4), round(float(arr.std(ddof=1)), 4),
                   len(vals), [round(v, 4) for v in vals])


def proportion_ci(successes: int, n: int,
                  confidence: float = config.CONFIDENCE_LEVEL,
                  ) -> tuple[float, float] | None:
    """Wilson score interval for a proportion.

    Wilson rather than normal-approximation because gold sets are small and
    several rates here (repealed-citation rate, refusal recall) sit near 0 or 1,
    where the normal approximation produces intervals that run outside [0, 1].

    Uses `scipy.stats.norm.ppf` for the normal quantile (lazy import -- see
    `eval/gold_sets.py::proportion_ci_halfwidth`'s docstring for why this
    replaced a hand-rolled z-table during the audit, docs/M6_GAP_ANALYSIS.md G3).
    """
    if n <= 0:
        return None
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)
