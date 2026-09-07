"""Typed schemas for the five gold sets and the annotation record.

One definition per set, used by the annotation app, the gold-set loader, the
runner and the tests -- so a field name can never drift between the tool that
writes a row and the code that reads it.

Field naming deliberately mirrors `smartlawai.protocols.Claim`
(`text` / `passage_ids` / `citations`) so `run_eval.py` can compare a real
`GenerationResult` against gold data without a translation layer
(M6_ONBOARDING.md §12).

Imports nothing from `smartlawai`: pure data structures (OPS.md §8).

`groundedness.jsonl` and `entailment.jsonl` share a shape but are NOT the same
set and must never be merged -- see GOLD_SET_SPECS for what separates them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar

from eval import config

# --------------------------------------------------------------------------- #
# Label domains -- enumerated in code, not prose, so validation is real
# --------------------------------------------------------------------------- #
OOS_LABELS: tuple[str, ...] = ("in_scope", "out_of_scope", "advice_seeking")
ENTAILMENT_LABELS: tuple[str, ...] = ("entailed", "not_entailed", "unclear")
RELEVANCE_GRADES: tuple[int, ...] = (0, 1, 2)  # TREC-style: not / partially / highly
IN_FORCE_LABELS: tuple[str, ...] = ("in_force", "repealed", "unknown")
ANNOTATION_MODES: tuple[str, ...] = ("correction", "cold")


class SchemaError(ValueError):
    """A gold row that does not satisfy its schema. Carries the offending line."""


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
@dataclass
class GoldItem:
    """Fields every gold row carries, whatever the set.

    `source_doc_id` exists so M1's planned `assert_no_eval_leakage()` helper
    (M1_ONBOARDING.md §9) works against these files unchanged the moment it
    lands -- the field name is theirs, not ours.
    """

    item_id: str
    source_doc_id: str = ""
    split: str = ""            # "" until data/splits/*.json exists (I5)
    difficulty: str = ""       # "easy" | "medium" | "hard" -- free text, optional
    category: str = ""
    notes: str = ""
    # Annotation provenance. Deliberately pool-agnostic: rater_ids are opaque
    # strings, never "student"/"advocate" (PLAN.md M6: annotator-pool-agnostic).
    rater_ids: list[str] = field(default_factory=list)
    n_annotations: int = 0
    agreement: float | None = None   # None = not measured, never a filled-in 0.0
    rubric_version: str = ""
    provenance: str = ""             # how this row came to exist, in one line

    REQUIRED: ClassVar[tuple[str, ...]] = ("item_id",)

    def validate(self) -> None:
        for name in self.REQUIRED:
            value = getattr(self, name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise SchemaError(f"{type(self).__name__}: missing required field {name!r}")
            if isinstance(value, (list, tuple, dict)) and len(value) == 0:
                raise SchemaError(f"{type(self).__name__}: field {name!r} must be non-empty")

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict[str, Any]):
        known = {f.name for f in fields(cls)}
        unknown = set(row) - known
        if unknown:
            raise SchemaError(
                f"{cls.__name__}: unknown field(s) {sorted(unknown)}; "
                f"known fields are {sorted(known)}")
        obj = cls(**{k: v for k, v in row.items() if k in known})
        obj.validate()
        return obj


# --------------------------------------------------------------------------- #
# 1. retrieval.jsonl -- TREC pooling, graded relevance
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalItem(GoldItem):
    """One query plus its judged pool.

    `judgments` maps chunk_id -> relevance grade (RELEVANCE_GRADES). Only the
    ~15 pooled candidates are judged (RESEARCH.md §7.1.2); an unjudged chunk is
    absent from the dict, which is NOT the same as a judged-irrelevant 0 and is
    treated differently by nDCG.
    """

    query: str = ""
    doc_ids: list[str] = field(default_factory=list)
    judgments: dict[str, int] = field(default_factory=dict)
    pooled_from: list[str] = field(default_factory=list)  # which retrievers contributed

    REQUIRED: ClassVar[tuple[str, ...]] = ("item_id", "query")

    def validate(self) -> None:
        super().validate()
        for chunk_id, grade in self.judgments.items():
            if not isinstance(grade, int) or grade not in RELEVANCE_GRADES:
                raise SchemaError(
                    f"RetrievalItem {self.item_id}: judgment for {chunk_id!r} is "
                    f"{grade!r}; must be one of {RELEVANCE_GRADES}")

    def relevant_ids(self, min_grade: int = 1) -> set[str]:
        return {cid for cid, g in self.judgments.items() if g >= min_grade}

    def gains(self) -> dict[str, float]:
        """Graded gains for nDCG. Grade 0 rows are judged-irrelevant, so they
        stay out of the ideal ranking."""
        return {cid: float(g) for cid, g in self.judgments.items() if g > 0}


# --------------------------------------------------------------------------- #
# 2. oos.jsonl -- three-way scope classification
# --------------------------------------------------------------------------- #
@dataclass
class OOSItem(GoldItem):
    """A question labelled in_scope / out_of_scope / advice_seeking.

    Feeds M5's OOS threshold calibration (PLAN.md M5.6). M6 produces the
    labelled set; it calibrates nothing.
    """

    question: str = ""
    label: str = ""
    doc_ids: list[str] = field(default_factory=list)

    REQUIRED: ClassVar[tuple[str, ...]] = ("item_id", "question", "label")

    def validate(self) -> None:
        super().validate()
        if self.label not in OOS_LABELS:
            raise SchemaError(
                f"OOSItem {self.item_id}: label {self.label!r} not in {OOS_LABELS}")

    def should_refuse(self) -> bool:
        return self.label in ("out_of_scope", "advice_seeking")


# --------------------------------------------------------------------------- #
# 3 & 4. groundedness.jsonl and entailment.jsonl -- (passage, claim, label)
# --------------------------------------------------------------------------- #
@dataclass
class ClaimPassageItem(GoldItem):
    """Shared shape for the two (passage, claim, label) sets.

    `claim_text` / `passage_ids` / `citations` mirror protocols.Claim so a real
    GenerationResult drops straight in once M3 lands.
    """

    claim_text: str = ""
    passage: str = ""
    passage_ids: list[str] = field(default_factory=list)
    label: str = ""
    citations: list[str] = field(default_factory=list)

    REQUIRED: ClassVar[tuple[str, ...]] = ("item_id", "claim_text", "passage", "label")

    def validate(self) -> None:
        super().validate()
        if self.label not in ENTAILMENT_LABELS:
            raise SchemaError(
                f"{type(self).__name__} {self.item_id}: label {self.label!r} "
                f"not in {ENTAILMENT_LABELS}")

    def supported(self) -> bool | None:
        """True/False, or None for 'unclear' -- an unclear human judgment is not
        evidence of a hallucination and must not be scored as one (I2)."""
        if self.label == "unclear":
            return None
        return self.label == "entailed"


@dataclass
class GroundednessItem(ClaimPassageItem):
    """EVALUATION set: scores the running system's own claim/passage pairs."""


@dataclass
class EntailmentItem(ClaimPassageItem):
    """HELD-OUT JUDGE-EVALUATION set: measures how good InLegalNLI / HHEM are as
    classifiers.

    RESEARCH.md §2.5 hard rule: **never used to train InLegalNLI.** Kept as a
    separate file with separate provenance from groundedness.jsonl even though
    the schema is identical, precisely so nobody merges them by accident.
    `tests/test_perturbations.py` enforces that no perturbation training row is
    derived from a document appearing here.
    """

    judge_source: str = ""  # e.g. "internal" | "claimrag-law" -- external rows stay tagged


# --------------------------------------------------------------------------- #
# 5. repealed.jsonl -- the IPC -> BNS gold set
# --------------------------------------------------------------------------- #
@dataclass
class RepealedItem(GoldItem):
    """A criminal-law query plus the objectively correct authority status.

    0% duplicated for IAA (RESEARCH.md §7.1.3): it is a registry lookup, and the
    objectivity is the point. Release-ready from the start (PLAN.md M6) --
    publishing this as a standalone benchmark should be a licensing decision,
    not rework, so every row carries its own authority provenance.
    """

    question: str = ""
    citation: str = ""              # e.g. "IPC s.420"
    in_force_label: str = ""        # IN_FORCE_LABELS
    superseded_by: str = ""         # e.g. "BNS s.318" -- "" when still in force
    as_of_date: str = ""            # ISO date the judgment is made as of
    authority_source: str = ""      # primary source: Gazette of India, eSCR, ...

    REQUIRED: ClassVar[tuple[str, ...]] = ("item_id", "question", "citation",
                                           "in_force_label")

    def validate(self) -> None:
        super().validate()
        if self.in_force_label not in IN_FORCE_LABELS:
            raise SchemaError(
                f"RepealedItem {self.item_id}: in_force_label "
                f"{self.in_force_label!r} not in {IN_FORCE_LABELS}")
        if self.in_force_label == "repealed" and not self.superseded_by:
            raise SchemaError(
                f"RepealedItem {self.item_id}: a repealed provision must name what "
                f"superseded it (or 'none' if genuinely nothing did)")

    def as_registry_result(self) -> dict[str, Any]:
        """Shape `metrics.repealed_citation_rate()` consumes -- M4's OWN
        published `RegistryResult` contract (M4_ONBOARDING.md §6:
        `{status: "in_force"|"repealed"|"superseded"|"not_found", as_of, note}`),
        reused verbatim rather than an invented shape, so a real M4 result
        drops in with no translation layer the day that module lands.

        This is a fixture for TESTING the metric functions against known-good
        input, not a stand-in used in a live `run_eval.py` run -- M6_ONBOARDING
        §8 is explicit that citation_validity_rate/repealed_citation_rate
        "can't be computed against a real registry until M4 lands", so the
        harness reports them `unavailable` in a real run rather than feeding
        them this gold label reinterpreted as a fake registry answer.
        """
        status = {"in_force": "in_force", "repealed": "repealed",
                  "unknown": "not_found"}[self.in_force_label]
        note = (f"repealed; see {self.superseded_by}" if self.superseded_by
               else None)
        return {"citation": self.citation, "status": status,
                "as_of": self.as_of_date or None, "note": note}


# --------------------------------------------------------------------------- #
# Annotation record -- what the app appends on every submission
# --------------------------------------------------------------------------- #
@dataclass
class Annotation:
    """One rater's judgment of one item.

    Records exactly what RESEARCH.md §7.2 and PLAN.md M6 require: rater id,
    timestamp, **time per item** (the number that sizes every remaining gold
    set), and whether the label was given cold or corrected from a model
    proposal (the anchoring control).
    """

    annotation_id: str
    task: str                  # gold set name: retrieval | oos | groundedness | ...
    item_id: str
    rater_id: str
    label: Any                 # str for classification, dict for graded pools
    mode: str                  # ANNOTATION_MODES
    created_at: str            # ISO-8601 UTC
    seconds_on_item: float
    proposed_label: Any = None  # what the model suggested in correction mode
    accepted_proposal: bool | None = None  # None in cold mode -- nothing to accept
    confidence: int | None = None          # optional 1-5 self-report
    comment: str = ""
    rubric_version: str = ""
    session_id: str = ""

    def validate(self) -> None:
        if not self.item_id or not self.rater_id or not self.task:
            raise SchemaError("Annotation: task, item_id and rater_id are required")
        if self.mode not in ANNOTATION_MODES:
            raise SchemaError(
                f"Annotation {self.annotation_id}: mode {self.mode!r} not in "
                f"{ANNOTATION_MODES}")
        if self.seconds_on_item < 0:
            raise SchemaError("Annotation: seconds_on_item cannot be negative")
        if self.mode == "cold" and self.proposed_label is not None:
            raise SchemaError(
                "Annotation: cold mode must not carry a proposed_label -- that "
                "would defeat the anchoring control (RESEARCH.md §7.1.1)")

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Annotation:
        known = {f.name for f in fields(cls)}
        obj = cls(**{k: v for k, v in row.items() if k in known})
        obj.validate()
        return obj


# --------------------------------------------------------------------------- #
# The registry of gold sets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GoldSetSpec:
    name: str
    filename: str
    schema: type
    labels: tuple[str, ...]
    iaa_duplication_rate: float   # RESEARCH.md §7.1.3
    target_n: int                 # RESEARCH.md §7.1.4, stated not assumed
    purpose: str
    never_train_on: bool = False


GOLD_SET_SPECS: dict[str, GoldSetSpec] = {
    "retrieval": GoldSetSpec(
        name="retrieval", filename="retrieval.jsonl", schema=RetrievalItem,
        labels=tuple(str(g) for g in RELEVANCE_GRADES),
        iaa_duplication_rate=config.IAA_DUPLICATION_RATE["retrieval"],
        target_n=config.TARGET_N["retrieval"],
        purpose="Is this chunk relevant to this query? TREC-pooled candidates."),
    "oos": GoldSetSpec(
        name="oos", filename="oos.jsonl", schema=OOSItem, labels=OOS_LABELS,
        iaa_duplication_rate=config.IAA_DUPLICATION_RATE["oos"],
        target_n=config.TARGET_N["oos"],
        purpose="In-scope / out-of-scope / advice-seeking. Feeds M5's OOS threshold."),
    "groundedness": GoldSetSpec(
        name="groundedness", filename="groundedness.jsonl", schema=GroundednessItem,
        labels=ENTAILMENT_LABELS,
        iaa_duplication_rate=config.IAA_DUPLICATION_RATE["groundedness"],
        target_n=config.TARGET_N["groundedness"],
        purpose="Is this claim supported by its cited passage? Scores the system."),
    "repealed": GoldSetSpec(
        name="repealed", filename="repealed.jsonl", schema=RepealedItem,
        labels=IN_FORCE_LABELS,
        iaa_duplication_rate=config.IAA_DUPLICATION_RATE["repealed"],
        target_n=config.TARGET_N["repealed"],
        purpose="Criminal-law queries for the IPC->BNS natural experiment."),
    "entailment": GoldSetSpec(
        name="entailment", filename="entailment.jsonl", schema=EntailmentItem,
        labels=ENTAILMENT_LABELS,
        iaa_duplication_rate=config.IAA_DUPLICATION_RATE["entailment"],
        target_n=config.TARGET_N["entailment"],
        purpose="Held-out judge-evaluation set. Scores InLegalNLI/HHEM as classifiers.",
        never_train_on=True),
}

GOLD_SET_NAMES: tuple[str, ...] = tuple(GOLD_SET_SPECS)


def validate_row(task: str, row: dict[str, Any]):
    """Parse and validate one raw JSONL row against its set's schema."""
    if task not in GOLD_SET_SPECS:
        raise SchemaError(f"unknown gold set {task!r}; known: {GOLD_SET_NAMES}")
    return GOLD_SET_SPECS[task].schema.from_row(row)
