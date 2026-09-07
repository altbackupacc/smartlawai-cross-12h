"""Every threshold, path and constant the M6 harness uses.

CLAUDE.md §4 ("no magic numbers outside config.py") applied to the eval package.
Deliberately separate from src/smartlawai/config.py: those are *serving*
thresholds that ship in the product; these are *measurement* choices that belong
to the paper. Mixing them would put eval concerns inside serving code.

Every value below traces to a line in RESEARCH.md or PLAN.md -- the citation is
part of the constant, not a comment someone can drift away from.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
EVAL_DIR: Path = REPO_ROOT / "eval"
GOLD_DIR: Path = EVAL_DIR / "gold"
EXTERNAL_GOLD_DIR: Path = GOLD_DIR / "external"
RESULTS_DIR: Path = EVAL_DIR / "results"
ANNOTATIONS_DIR: Path = EVAL_DIR / "annotations"
SPLITS_DIR: Path = REPO_ROOT / "data" / "splits"
PERTURBATIONS_DIR: Path = REPO_ROOT / "data" / "perturbations"

# IN-Abs plain-text corpus, already pulled by M1 (M6_ONBOARDING.md §12: "don't
# re-pull it"). Text files, so the perturbation generator needs no `datasets`.
IN_ABS_DIR: Path = (REPO_ROOT / "data" / "raw" / "in_ext_source" / "dataset" / "IN-Abs")

# --------------------------------------------------------------------------- #
# Statistical protocol (RESEARCH.md §6)
# --------------------------------------------------------------------------- #
SEEDS: tuple[int, ...] = (42, 1337, 2024, 31337, 8191)  # pre-registered
DEFAULT_SEED: int = 42
BOOTSTRAP_RESAMPLES: int = 10_000
CONFIDENCE_LEVEL: float = 0.95
ALPHA: float = 0.05
ECE_BINS: int = 10

# --------------------------------------------------------------------------- #
# Metric parameters (RESEARCH.md §5)
# --------------------------------------------------------------------------- #
RETRIEVAL_K_VALUES: tuple[int, ...] = (5, 10, 20)  # "Recall@{5,10,20}"
NDCG_K: int = 10                                    # "nDCG@10"
EXTRACTIVENESS_NGRAM: int = 4                       # "4-gram overlap with source"
ROUGE_VARIANTS: tuple[str, ...] = ("rouge1", "rouge2", "rougeL")

# --------------------------------------------------------------------------- #
# Annotation protocol (RESEARCH.md §7, PLAN.md M6)
# --------------------------------------------------------------------------- #
PILOT_SIZE: int = 20            # §7.3 "always pilot 20 items first"
COLD_SUBSET_FRACTION: float = 0.10  # §7.1.1 anchoring control
POOL_DEPTH: int = 10            # §7.1.2 TREC pooling: top-10 per method
POOL_METHODS: tuple[str, ...] = ("bm25", "dense", "hybrid")

# §7.1.3 "duplicate only where subjectivity lives" -- the fraction of each set
# that gets a second independent rater so an agreement statistic exists.
IAA_DUPLICATION_RATE: dict[str, float] = {
    "retrieval": 1.00,
    "groundedness": 1.00,
    "entailment": 1.00,
    "oos": 0.30,
    "repealed": 0.00,  # objective registry lookup -- duplication buys nothing
}

# §7.1.4 "right-size with a power calculation": n~=300 gives ~+/-5% on a
# proportion. The number below is the target; the achieved n and its actual CI
# half-width are computed from the file, never assumed (see gold_sets.py).
TARGET_N: dict[str, int] = {
    "retrieval": 200,    # queries, not judgments (~15 pooled candidates each)
    "oos": 350,
    "groundedness": 350,
    "repealed": 350,
    "entailment": 350,
}

# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
EVAL_OWNER_ID: str = "eval-harness"  # I1: Scope needs an owner; eval gets its own
SPLITS: tuple[str, ...] = ("train", "dev", "test")

# CI eval-regression gate (M6_ONBOARDING.md §11). Deliberately generous: with
# every model stubbed this guards against the harness silently breaking, not
# against quality regressions that cannot exist yet. Tighten as real components
# land -- that is a later milestone's call, not a prediction to make now.
REGRESSION_TOLERANCE: float = 0.10

# --------------------------------------------------------------------------- #
# Perturbation generator (RESEARCH.md §2.5)
# --------------------------------------------------------------------------- #
PERTURBATION_TYPES: tuple[str, ...] = (
    "section_number",       # wrong statutory provision
    "ipc_bns_swap",         # repealed / superseded law
    "negate_obligation",    # reversed legal effect ("shall" -> "shall not")
    "swap_parties",         # misattributed holding
    "alter_date_amount",    # factual error
    "change_court",         # wrong precedential weight
    "unsupported_holding",  # fabricated ratio
)
