"""Synthetic legal-perturbation generator for InLegalNLI training data (M8a).

Built under M6 rather than M8a for one reason (M6_ONBOARDING.md §9): it must be
written with evaluation-split safety designed in from day one, and M6 is the
milestone that owns what "an evaluation split" means here.

  * This is TRAINING data for a different milestone's model. It is not
    evaluation data, and nothing in eval/ scores against it.
  * Releasable standalone -- RESEARCH.md §2.5's hard rules require the
    generator itself to be published, not just its output. It therefore has no
    imports outside the standard library and no dependency on this repo's
    packages: copy this file plus the IN-Abs corpus and it runs.
  * Every row is tagged with its source `doc_id`, so disjointness from every
    evaluation split is checkable (tests/test_perturbations.py).
  * Sources are genuine IN-Abs (passage, claim) pairs, never this project's own
    system outputs -- otherwise the judge learns to like our writing style
    instead of detecting faithfulness failures (RESEARCH.md §2.5).

The seven perturbation types are RESEARCH.md §2.5's taxonomy verbatim:

  section_number       wrong statutory provision
  ipc_bns_swap         repealed / superseded law
  negate_obligation    reversed legal effect ("shall" -> "shall not")
  swap_parties         misattributed holding
  alter_date_amount    factual error
  change_court         wrong precedential weight
  unsupported_holding  fabricated ratio

How genuinely-entailed source pairs are obtained, stated plainly because it is a
methodological choice a reviewer will ask about: IN-Abs ships each judgment with
a professionally written headnote. A headnote sentence is treated as a candidate
claim, and it is paired with the judgment window that shares the most content
words with it. A pair is kept only when that overlap clears `--min-overlap`,
which is a proxy for entailment, not a proof of it. Every row records its
overlap score so a stricter filter can be applied downstream without
regenerating, and `--min-overlap` is recorded in the run manifest.

Usage:
    python data/perturbations/generate_perturbations.py --n 500
    python data/perturbations/generate_perturbations.py --n 50 --seed 1337 \
        --out data/perturbations/perturbations.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

GENERATOR_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS = REPO_ROOT / "data" / "raw" / "in_ext_source" / "dataset" / "IN-Abs"
DEFAULT_OUT = REPO_ROOT / "data" / "perturbations" / "perturbations.jsonl"

PERTURBATION_TYPES = (
    "section_number", "ipc_bns_swap", "negate_obligation", "swap_parties",
    "alter_date_amount", "change_court", "unsupported_holding",
)

# Passage window, in sentences, around the best-matching judgment sentence.
PASSAGE_WINDOW = 3
MIN_CLAIM_WORDS = 8
MAX_CLAIM_WORDS = 60
DEFAULT_MIN_OVERLAP = 0.35

_SENTENCE = re.compile(r"(?<=[.;:])\s+|\n+")
_WORD = re.compile(r"[a-z]{3,}")

# Words that carry no discriminating content when matching a headnote sentence
# to its judgment passage.
_STOP = frozenset(["the", "and", "that", "for", "with", "was", "were", "has", "have", "had", "been", "are", "this", "which", "their", "from", "not", "but", "any", "all", "such", "under", "upon", "shall", "may", "can", "will", "would", "could", "should", "his", "her", "its", "they", "them", "then", "than", "when", "what", "who", "whom", "whose", "out", "into", "about", "also", "more", "most", "other", "some", "only", "very", "said", "state", "court", "case", "appeal"])


# --------------------------------------------------------------------------- #
# Row schema
# --------------------------------------------------------------------------- #
@dataclass
class PerturbationRow:
    pair_id: str
    source_doc_id: str
    source_corpus: str
    source_split: str          # the CORPUS split; "" means data/splits/*.json absent
    perturbation_type: str     # "none" for the unperturbed positive
    label: str                 # "entailed" | "not_entailed"
    passage: str
    claim: str
    original_claim: str
    overlap: float
    seed: int
    generator_version: str = GENERATOR_VERSION
    detail: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# The seven perturbations. Each returns (perturbed_claim, detail) or None when
# the type does not apply to this claim -- a type that cannot fire is skipped,
# never faked, so per-type detection rates measure real instances only.
# --------------------------------------------------------------------------- #
_SECTION = re.compile(r"\b(section|sections|s\.|ss\.)\s*(\d+)([A-Za-z]?)\b", re.IGNORECASE)


def perturb_section_number(claim: str, rng: random.Random):
    """Wrong statutory provision: change a cited section number."""
    matches = list(_SECTION.finditer(claim))
    if not matches:
        return None
    m = rng.choice(matches)
    original = int(m.group(2))
    delta = rng.choice([d for d in (-100, -11, -3, -1, 1, 3, 11, 100)
                        if original + d > 0])
    replacement = f"{m.group(1)} {original + delta}{m.group(3)}"
    perturbed = claim[:m.start()] + replacement + claim[m.end():]
    return perturbed, {"from": m.group(0), "to": replacement}


# IPC -> BNS is not a renumbering with a formula; these are the mappings the
# IPC->BNS experiment actually cares about (RESEARCH.md §2). The perturbation
# is the INCORRECT swap: keep the section number while changing the statute
# name, which is precisely the error a model trained on pre-2024 text makes.
_IPC_NAMES = ("Indian Penal Code", "I.P.C.", "IPC", "Penal Code")
_BNS_NAMES = ("Bharatiya Nyaya Sanhita", "BNS")
_CRPC_NAMES = ("Code of Criminal Procedure", "Cr.P.C.", "CrPC")
_BNSS_NAMES = ("Bharatiya Nagarik Suraksha Sanhita", "BNSS")


def perturb_ipc_bns_swap(claim: str, rng: random.Random):
    """Repealed / superseded law: rename the statute without renumbering it."""
    for names, replacements in ((_IPC_NAMES, _BNS_NAMES), (_CRPC_NAMES, _BNSS_NAMES),
                                (_BNS_NAMES, _IPC_NAMES), (_BNSS_NAMES, _CRPC_NAMES)):
        for name in names:
            pattern = re.compile(re.escape(name), re.IGNORECASE)
            if pattern.search(claim):
                replacement = rng.choice(replacements)
                return (pattern.sub(replacement, claim, count=1),
                        {"from": name, "to": replacement})
    return None


_NEGATIONS: tuple[tuple[str, str], ...] = (
    (r"\bshall not\b", "shall"),
    (r"\bmust not\b", "must"),
    (r"\bcannot\b", "can"),
    (r"\bis not entitled\b", "is entitled"),
    (r"\bwas not entitled\b", "was entitled"),
    (r"\bshall\b", "shall not"),
    (r"\bmust\b", "must not"),
    (r"\bis entitled\b", "is not entitled"),
    (r"\bwas entitled\b", "was not entitled"),
    (r"\bis liable\b", "is not liable"),
    (r"\bwas liable\b", "was not liable"),
    (r"\bis barred\b", "is not barred"),
)


def perturb_negate_obligation(claim: str, rng: random.Random):
    """Reversed legal effect. Negation-removals are tried before
    negation-insertions so 'shall not' never becomes 'shall not not'."""
    for pattern, replacement in _NEGATIONS:
        compiled = re.compile(pattern, re.IGNORECASE)
        if compiled.search(claim):
            return (compiled.sub(replacement, claim, count=1),
                    {"from": pattern, "to": replacement})
    return None


_PARTY_PAIRS: tuple[tuple[str, str], ...] = (
    ("appellant", "respondent"), ("plaintiff", "defendant"),
    ("petitioner", "respondent"), ("appellants", "respondents"),
    ("plaintiffs", "defendants"), ("assessee", "revenue"),
    ("accused", "complainant"), ("landlord", "tenant"),
)


def _swap_words(text: str, a: str, b: str) -> tuple[str, int]:
    """Swap two words case-insensitively while preserving each occurrence's case."""
    n = 0

    def preserve(source: str, target: str) -> str:
        if source.isupper():
            return target.upper()
        if source[:1].isupper():
            return target.capitalize()
        return target

    def repl(m: re.Match) -> str:
        nonlocal n
        n += 1
        word = m.group(0)
        other = b if word.lower() == a.lower() else a
        return preserve(word, other)

    pattern = re.compile(rf"\b({re.escape(a)}|{re.escape(b)})\b", re.IGNORECASE)
    return pattern.sub(repl, text), n


def perturb_swap_parties(claim: str, rng: random.Random):
    """Misattributed holding: swap who won."""
    applicable = [(a, b) for a, b in _PARTY_PAIRS
                  if re.search(rf"\b({re.escape(a)}|{re.escape(b)})\b", claim, re.IGNORECASE)]
    if not applicable:
        return None
    a, b = rng.choice(applicable)
    perturbed, n = _swap_words(claim, a, b)
    if n == 0 or perturbed == claim:
        return None
    return perturbed, {"pair": [a, b], "n_swapped": n}


_YEAR = re.compile(r"\b(1[89]\d{2}|20[0-4]\d)\b")
_AMOUNT = re.compile(r"\b(?:Rs\.?|rupees)\s*([\d,]+)", re.IGNORECASE)
_ORDINAL_DATE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)


def perturb_alter_date_amount(claim: str, rng: random.Random):
    """Factual error: move a year, a money figure, or a day-of-month."""
    amounts = list(_AMOUNT.finditer(claim))
    if amounts:
        m = rng.choice(amounts)
        raw = m.group(1).replace(",", "")
        if raw.isdigit() and int(raw) > 0:
            factor = rng.choice([0.1, 0.5, 2, 10])
            new_value = max(1, int(int(raw) * factor))
            perturbed = claim[:m.start(1)] + f"{new_value:,}" + claim[m.end(1):]
            return perturbed, {"kind": "amount", "from": m.group(1),
                               "to": f"{new_value:,}"}

    years = list(_YEAR.finditer(claim))
    if years:
        m = rng.choice(years)
        delta = rng.choice([-17, -9, -5, -2, 2, 5, 9, 17])
        new_year = int(m.group(1)) + delta
        perturbed = claim[:m.start()] + str(new_year) + claim[m.end():]
        return perturbed, {"kind": "year", "from": m.group(1), "to": str(new_year)}

    days = list(_ORDINAL_DATE.finditer(claim))
    if days:
        m = rng.choice(days)
        original = int(m.group(1))
        new_day = original % 28 + 1
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(
            new_day if new_day < 20 else new_day % 10, "th")
        perturbed = claim[:m.start()] + f"{new_day}{suffix}" + claim[m.end():]
        return perturbed, {"kind": "day", "from": m.group(0),
                           "to": f"{new_day}{suffix}"}
    return None


_COURTS: tuple[str, ...] = (
    "Supreme Court", "High Court", "Bombay High Court", "Calcutta High Court",
    "Madras High Court", "Delhi High Court", "Allahabad High Court",
    "District Court", "Privy Council", "Sessions Court",
)


def perturb_change_court(claim: str, rng: random.Random):
    """Wrong precedential weight: attribute the holding to a different court."""
    present = [c for c in _COURTS if re.search(re.escape(c), claim, re.IGNORECASE)]
    if not present:
        return None
    # Longest match first, so "Bombay High Court" is not clobbered by "High Court".
    original = max(present, key=len)
    replacement = rng.choice([c for c in _COURTS if c.lower() != original.lower()])
    perturbed = re.sub(re.escape(original), replacement, claim, count=1,
                       flags=re.IGNORECASE)
    return perturbed, {"from": original, "to": replacement}


_FABRICATED_HOLDINGS: tuple[str, ...] = (
    "The Court further held that the limitation period stood extended by two years.",
    "It was also held that the burden of proof lay entirely on the respondent.",
    "The Court additionally ruled that no appeal shall lie against this finding.",
    "It was further observed that the impugned order was void ab initio.",
    "The Court also held that costs were awarded against the appellant throughout.",
    "It was additionally held that the statutory notice requirement stood waived.",
)


def perturb_unsupported_holding(claim: str, rng: random.Random):
    """Fabricated ratio: append a holding the passage never establishes.

    The only always-applicable type, so it is placed last in the try-order and
    the generator balances type counts to stop it dominating the output.
    """
    fabricated = rng.choice(_FABRICATED_HOLDINGS)
    separator = " " if claim.rstrip().endswith((".", ";", ":")) else ". "
    return claim.rstrip() + separator + fabricated, {"inserted": fabricated}


PERTURBERS: dict[str, Callable[[str, random.Random], tuple[str, dict] | None]] = {
    "section_number": perturb_section_number,
    "ipc_bns_swap": perturb_ipc_bns_swap,
    "negate_obligation": perturb_negate_obligation,
    "swap_parties": perturb_swap_parties,
    "alter_date_amount": perturb_alter_date_amount,
    "change_court": perturb_change_court,
    "unsupported_holding": perturb_unsupported_holding,
}


# --------------------------------------------------------------------------- #
# Source pair extraction
# --------------------------------------------------------------------------- #
def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s and s.strip()]


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def overlap_score(claim: str, passage: str) -> float:
    """Fraction of the claim's content words present in the passage.

    A proxy for entailment, not a proof of one -- stated in the module docstring
    and recorded per row so it can be filtered harder later.
    """
    claim_words = content_words(claim)
    if not claim_words:
        return 0.0
    return len(claim_words & content_words(passage)) / len(claim_words)


@dataclass
class SourcePair:
    doc_id: str
    claim: str
    passage: str
    overlap: float


def iter_documents(corpus_dir: Path, split_dir: str) -> Iterator[tuple[str, str, str]]:
    """Yield (doc_id, judgment_text, summary_text) from an IN-Abs split folder."""
    judgements = corpus_dir / split_dir / "judgement"
    summaries = corpus_dir / split_dir / "summary"
    if not judgements.is_dir():
        raise FileNotFoundError(
            f"IN-Abs judgements not found at {judgements}. Pass --corpus, or pull "
            f"the corpus first (M1 owns data/raw/; do not re-pull it if it exists).")
    for path in sorted(judgements.glob("*.txt"), key=lambda p: p.stem):
        summary_path = summaries / path.name
        if not summary_path.exists():
            continue
        yield (path.stem,
               path.read_text(encoding="utf-8", errors="ignore"),
               summary_path.read_text(encoding="utf-8", errors="ignore"))


def extract_pairs(doc_id: str, judgment: str, summary: str,
                  min_overlap: float = DEFAULT_MIN_OVERLAP,
                  max_pairs_per_doc: int = 3) -> list[SourcePair]:
    """Pair each headnote sentence with its best-matching judgment window."""
    judgment_sentences = split_sentences(judgment)
    if not judgment_sentences:
        return []
    sentence_words = [content_words(s) for s in judgment_sentences]

    pairs: list[SourcePair] = []
    for claim in split_sentences(summary):
        n_words = len(claim.split())
        if not MIN_CLAIM_WORDS <= n_words <= MAX_CLAIM_WORDS:
            continue
        claim_words = content_words(claim)
        if not claim_words:
            continue

        best_index, best_hits = -1, 0
        for i, words in enumerate(sentence_words):
            hits = len(claim_words & words)
            if hits > best_hits:
                best_index, best_hits = i, hits
        if best_index < 0:
            continue

        low = max(0, best_index - PASSAGE_WINDOW // 2)
        passage = " ".join(judgment_sentences[low:low + PASSAGE_WINDOW])
        score = overlap_score(claim, passage)
        if score >= min_overlap:
            pairs.append(SourcePair(doc_id, claim, passage, round(score, 4)))
        if len(pairs) >= max_pairs_per_doc:
            break
    return pairs


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def load_split_doc_ids(splits_dir: Path, split: str) -> set[str] | None:
    """Read data/splits/<split>.json if it exists; None means it does not.

    None is not an error here: I5's frozen splits are M1's deliverable and may
    not have landed. The caller degrades to an explicit, stated skip rather than
    silently pretending every document is safe to use.
    """
    path = splits_dir / f"{split}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("doc_ids", "docs", "ids"):
            if key in data:
                return {str(x) for x in data[key]}
        return {str(k) for k in data}
    return {str(x) for x in data}


def generate(out_path: Path = DEFAULT_OUT,
             corpus_dir: Path = DEFAULT_CORPUS,
             n: int = 500,
             seed: int = 42,
             min_overlap: float = DEFAULT_MIN_OVERLAP,
             corpus_split: str = "train-data",
             include_positives: bool = True,
             splits_dir: Path | None = None) -> dict:
    """Read genuinely-entailed (passage, claim) pairs, apply one perturbation
    type per output row, write perturbations.jsonl, return a run manifest.

    Perturbation types are assigned round-robin over the types that actually
    apply to each claim, so no single always-applicable type (notably
    `unsupported_holding`) dominates and the per-type detection rates in
    RESEARCH.md §5 have comparable sample sizes.

    When `include_positives` is set, each perturbed row is accompanied by its
    unperturbed source pair labelled `entailed` -- an NLI training set needs
    both classes, and emitting the positive from the same document keeps the
    pair minimally different, which is the point of contrastive perturbation.
    """
    rng = random.Random(seed)
    splits_dir = splits_dir or (REPO_ROOT / "data" / "splits")

    train_ids = load_split_doc_ids(splits_dir, "train")
    frozen_splits_available = train_ids is not None

    rows: list[PerturbationRow] = []
    type_counts: dict[str, int] = {t: 0 for t in PERTURBATION_TYPES}
    skipped_not_in_train = 0
    docs_used: set[str] = set()

    for doc_id, judgment, summary in iter_documents(corpus_dir, corpus_split):
        if len(rows) >= n:
            break
        # I5 / RESEARCH.md §2.5: once frozen splits exist, perturbations may only
        # be built from TRAINING documents. Building them from dev/test documents
        # would leak familiarity with those documents into a judge that is later
        # evaluated on them.
        if frozen_splits_available and doc_id not in train_ids:
            skipped_not_in_train += 1
            continue

        for pair in extract_pairs(doc_id, judgment, summary, min_overlap):
            if len(rows) >= n:
                break
            # Prefer the least-used applicable type -> balanced type counts.
            applicable: list[tuple[str, str, dict]] = []
            for ptype in PERTURBATION_TYPES:
                result = PERTURBERS[ptype](pair.claim, rng)
                if result is not None and result[0] != pair.claim:
                    applicable.append((ptype, result[0], result[1]))
            if not applicable:
                continue
            ptype, perturbed, detail = min(applicable, key=lambda x: type_counts[x[0]])
            type_counts[ptype] += 1
            docs_used.add(doc_id)

            index = len(rows)
            if include_positives:
                rows.append(PerturbationRow(
                    pair_id=f"pert-{index:06d}-pos", source_doc_id=doc_id,
                    source_corpus="IN-Abs", source_split=corpus_split,
                    perturbation_type="none", label="entailed",
                    passage=pair.passage, claim=pair.claim,
                    original_claim=pair.claim, overlap=pair.overlap, seed=seed))
            rows.append(PerturbationRow(
                pair_id=f"pert-{index:06d}-neg", source_doc_id=doc_id,
                source_corpus="IN-Abs", source_split=corpus_split,
                perturbation_type=ptype, label="not_entailed",
                passage=pair.passage, claim=perturbed,
                original_claim=pair.claim, overlap=pair.overlap, seed=seed,
                detail=detail))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_row(), ensure_ascii=False,
                                sort_keys=True) + "\n")

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "requested_n": n,
        "n_rows": len(rows),
        "n_positive": sum(1 for r in rows if r.label == "entailed"),
        "n_perturbed": sum(1 for r in rows if r.label == "not_entailed"),
        "n_source_documents": len(docs_used),
        "corpus": str(corpus_dir),
        "corpus_split": corpus_split,
        "min_overlap": min_overlap,
        "type_counts": type_counts,
        "frozen_splits_available": frozen_splits_available,
        "train_split_filter_applied": frozen_splits_available,
        "skipped_documents_not_in_train_split": skipped_not_in_train,
        "out_path": str(out_path),
    }
    if not frozen_splits_available:
        manifest["split_safety_note"] = (
            "data/splits/train.json does not exist yet (M1 has not frozen the "
            "splits). Rows were drawn from the whole IN-Abs training folder and "
            "the train-split-only restriction could NOT be enforced. Regenerate "
            "once splits land; tests/test_perturbations.py enforces the check "
            "automatically from that moment.")

    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                        help="IN-Abs root containing train-data/ and test-data/")
    parser.add_argument("--corpus-split", default="train-data",
                        choices=["train-data", "test-data"])
    parser.add_argument("--n", type=int, default=500, help="max rows to emit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-overlap", type=float, default=DEFAULT_MIN_OVERLAP,
                        help="entailment proxy threshold for keeping a source pair")
    parser.add_argument("--no-positives", action="store_true",
                        help="emit only perturbed (not_entailed) rows")
    args = parser.parse_args()

    manifest = generate(out_path=args.out, corpus_dir=args.corpus, n=args.n,
                        seed=args.seed, min_overlap=args.min_overlap,
                        corpus_split=args.corpus_split,
                        include_positives=not args.no_positives)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
