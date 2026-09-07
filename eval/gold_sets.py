"""Load, validate, checksum and describe the five gold sets.

Gold sets are frozen reference data. They get the same "commit it, don't
regenerate it silently" care as `data/splits/*.json` (I5-adjacent, and
M6_ONBOARDING.md §14 says so explicitly), which is why every load records a
sha256: a silent edit to a committed gold file is then detectable by diffing a
results file's manifest against the current checksum.

I2 discipline for missing data: a gold set that is absent raises
`GoldSetMissing` with the path it looked for. It does NOT return an empty list,
because an empty list flows downstream into a `0.0` that reads like a measured
result. "We did not measure this" and "the system scored zero" are different
claims and the harness never blurs them.

Imports nothing from `smartlawai` (OPS.md §8).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval import config
from eval.schemas import GOLD_SET_NAMES, GOLD_SET_SPECS, SchemaError, validate_row


class GoldSetMissing(FileNotFoundError):
    """A requested gold set file does not exist. Never silently an empty set."""


@dataclass
class GoldSetStats:
    """Everything a provenance note or results manifest needs to say about a set."""

    name: str
    path: str
    n: int
    sha256: str
    label_counts: dict[str, int] = field(default_factory=dict)
    n_with_duplicate_annotation: int = 0
    duplication_rate_achieved: float = 0.0
    duplication_rate_target: float = 0.0
    target_n: int = 0
    ci_halfwidth_at_p50: float | None = None
    mean_agreement: float | None = None
    n_with_agreement: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "path": self.path, "n": self.n, "sha256": self.sha256,
            "label_counts": self.label_counts,
            "n_with_duplicate_annotation": self.n_with_duplicate_annotation,
            "duplication_rate_achieved": self.duplication_rate_achieved,
            "duplication_rate_target": self.duplication_rate_target,
            "target_n": self.target_n,
            "ci_halfwidth_at_p50": self.ci_halfwidth_at_p50,
            "mean_agreement": self.mean_agreement,
            "n_with_agreement": self.n_with_agreement,
        }


# --------------------------------------------------------------------------- #
# Power calculation (RESEARCH.md §7.1.4)
# --------------------------------------------------------------------------- #
def proportion_ci_halfwidth(n: int, p: float = 0.5,
                            confidence: float = config.CONFIDENCE_LEVEL) -> float | None:
    """Half-width of the normal-approximation CI for a proportion at size `n`.

    This is the calculation RESEARCH.md §7.1.4 demands be *stated* rather than
    replaced with a round number: n~=300 gives ~+/-5.7pp at p=0.5, n~=500 gives
    ~+/-4.4pp, and the returns past that are visibly diminishing. Reported per
    set from its ACHIEVED n, so a set that fell short of target says so in
    numbers rather than in a footnote nobody reads.

    Returns None for n <= 0 -- there is no interval around nothing.

    Uses `scipy.stats.norm.ppf` for the two-sided normal quantile (lazy
    import, matching the metrics.py convention for rouge-score/bert-score) --
    RESEARCH.md §5's "never hand-rolled where a real library exists" applies
    here too, not only to ROUGE/BERTScore; a hardcoded {0.90:1.6449,...} table
    was the audit's G3 finding (docs/M6_GAP_ANALYSIS.md) and is corrected here.
    """
    if n <= 0:
        return None
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    return round(z * math.sqrt(p * (1 - p) / n), 4)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def gold_path(name: str, gold_dir: Path | None = None) -> Path:
    if name not in GOLD_SET_SPECS:
        raise SchemaError(f"unknown gold set {name!r}; known: {GOLD_SET_NAMES}")
    return (gold_dir or config.GOLD_DIR) / GOLD_SET_SPECS[name].filename


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gold_set(name: str, gold_dir: Path | None = None) -> list:
    """Read and validate one gold set. Raises rather than returning [].

    A malformed row names its own line number -- annotation files are edited by
    hand often enough that "row 3 of oos.jsonl" beats a bare stack trace.
    """
    path = gold_path(name, gold_dir)
    if not path.exists():
        raise GoldSetMissing(
            f"gold set {name!r} not found at {path}. It has not been collected yet; "
            f"run the annotation app (eval/annotate/app.py) and export to this path. "
            f"Not returning an empty set -- that would score as a measured 0.0.")

    items = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SchemaError(f"{path}:{lineno}: invalid JSON: {e}") from e
            try:
                items.append(validate_row(name, row))
            except SchemaError as e:
                raise SchemaError(f"{path}:{lineno}: {e}") from e
    return items


def load_all_gold_sets(gold_dir: Path | None = None,
                       ) -> tuple[dict[str, list], dict[str, str]]:
    """Load every set that exists.

    Returns (loaded, missing) where `missing` maps set name -> the stated reason
    it could not be loaded. The caller reports the missing ones as `unavailable`
    rather than pretending the run covered them.
    """
    loaded: dict[str, list] = {}
    missing: dict[str, str] = {}
    for name in GOLD_SET_NAMES:
        try:
            loaded[name] = load_gold_set(name, gold_dir)
        except GoldSetMissing as e:
            missing[name] = str(e)
    return loaded, missing


def write_gold_set(name: str, items: list, gold_dir: Path | None = None) -> Path:
    """Write a gold set as JSONL, validating every item first.

    Deterministic key order so a re-export produces a reviewable diff rather
    than a whole-file churn.
    """
    path = gold_path(name, gold_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in items:
        item.validate()
        rows.append(json.dumps(item.to_row(), ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Description
# --------------------------------------------------------------------------- #
def _label_of(item) -> str | None:
    for attr in ("label", "in_force_label"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value:
            return value
    judgments = getattr(item, "judgments", None)
    if isinstance(judgments, dict) and judgments:
        return None  # graded pools are summarised by grade below, not by one label
    return None


def describe_gold_set(name: str, items: list,
                      gold_dir: Path | None = None) -> GoldSetStats:
    """Statistics a provenance note and a results manifest both need."""
    spec = GOLD_SET_SPECS[name]
    path = gold_path(name, gold_dir)
    counts: dict[str, int] = {}
    for item in items:
        label = _label_of(item)
        if label is not None:
            counts[label] = counts.get(label, 0) + 1
        else:
            for grade in getattr(item, "judgments", {}).values():
                key = f"grade_{grade}"
                counts[key] = counts.get(key, 0) + 1

    duplicated = sum(1 for i in items if getattr(i, "n_annotations", 0) >= 2)
    agreements = [i.agreement for i in items if getattr(i, "agreement", None) is not None]

    try:
        ci_halfwidth = proportion_ci_halfwidth(len(items))
    except ImportError:
        # scipy backs this (see the function's docstring); degrade to an
        # explicit None rather than crash the whole run for a caller who
        # hasn't installed the [eval] extra -- report.py/sizing_note already
        # render a None half-width as "not computed", never as 0%.
        ci_halfwidth = None

    return GoldSetStats(
        name=name,
        path=str(path),
        n=len(items),
        sha256=_sha256(path) if path.exists() else "",
        label_counts=dict(sorted(counts.items())),
        n_with_duplicate_annotation=duplicated,
        duplication_rate_achieved=round(duplicated / len(items), 4) if items else 0.0,
        duplication_rate_target=spec.iaa_duplication_rate,
        target_n=spec.target_n,
        ci_halfwidth_at_p50=ci_halfwidth,
        mean_agreement=round(sum(agreements) / len(agreements), 4) if agreements else None,
        n_with_agreement=len(agreements),
    )


def sizing_note(stats: GoldSetStats) -> str:
    """One human-readable line stating the power calculation for a set.

    RESEARCH.md §7.1.4: state the calculation, not a round number. Written from
    the ACHIEVED n so a short set reports its real, wider interval.
    """
    if stats.n == 0:
        return (f"{stats.name}: 0 items -- no interval exists. Target was "
                f"{stats.target_n}.")
    verdict = "meets target" if stats.n >= stats.target_n else "BELOW target"
    if stats.ci_halfwidth_at_p50 is None:
        return (f"{stats.name}: n={stats.n} ({verdict}, target {stats.target_n}); "
                f"CI half-width not computed (scipy unavailable).")
    return (f"{stats.name}: n={stats.n} ({verdict}, target {stats.target_n}) gives a "
            f"95% CI half-width of +/-{stats.ci_halfwidth_at_p50:.1%} on a proportion "
            f"at p=0.5.")
