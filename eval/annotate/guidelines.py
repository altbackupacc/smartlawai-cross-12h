"""Annotation rubrics as data, not prose in a README nobody opens.

Reference image 2 shows "Guidelines & rating scales" as a panel inside the
annotation app. That is the right call for a reason RESEARCH.md §7.3 makes
explicit: the rubric gets revised after the 20-item pilot, and a revision that
lives in a document diverges from what raters actually saw. Here the rubric is
versioned data, rendered in the app, and stamped onto every annotation, so an
agreement statistic can always be traced to the exact wording that produced it.

`RUBRIC_VERSION` MUST be bumped whenever any label description changes. Mixing
labels collected under different rubric versions into one agreement number is a
methodological error; `store.py` records the version per annotation so that
mixing is at least visible.

Imports nothing from `smartlawai` (OPS.md §8).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from eval.schemas import (
    ENTAILMENT_LABELS,
    IN_FORCE_LABELS,
    OOS_LABELS,
    RELEVANCE_GRADES,
)

# Bump on ANY wording change below. Pilot revisions are expected (RESEARCH.md §7.3).
RUBRIC_VERSION = "v1.0.0"


@dataclass(frozen=True)
class LabelOption:
    value: str
    title: str
    description: str
    hotkey: str = ""          # one item per screen, keyboard-driven (PLAN.md M6)
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class Rubric:
    task: str
    version: str
    question: str             # the single question put to the rater
    instructions: str
    options: tuple[LabelOption, ...]
    scale_note: str = ""
    pitfalls: tuple[str, ...] = ()
    fields_shown: tuple[str, ...] = field(default=())

    def option(self, value: str) -> LabelOption | None:
        return next((o for o in self.options if o.value == value), None)

    def values(self) -> tuple[str, ...]:
        return tuple(o.value for o in self.options)


RETRIEVAL_RUBRIC = Rubric(
    task="retrieval",
    version=RUBRIC_VERSION,
    question="How relevant is this passage to the query?",
    instructions=(
        "Judge the passage on its own. Do not reward a passage for being adjacent "
        "to a relevant one, and do not penalise it for being poorly written. The "
        "question is only whether a lawyer answering this query would want to read "
        "this passage."
    ),
    options=(
        LabelOption("0", "Not relevant", "Answers nothing in the query. Shared "
                    "vocabulary is not relevance.", hotkey="0"),
        LabelOption("1", "Partially relevant", "Contains part of what is needed, or "
                    "context that materially helps, but not the answer itself.",
                    hotkey="1"),
        LabelOption("2", "Highly relevant", "Contains the substance of the answer, or "
                    "the provision/holding the query is asking about.", hotkey="2"),
    ),
    scale_note=f"Graded relevance {RELEVANCE_GRADES} (TREC-style), not binary.",
    pitfalls=(
        ("Only the pooled candidates are shown -- an unshown chunk is unjudged, not "
        "irrelevant (RESEARCH.md §7.1.2)."),
        "Judge relevance to the QUERY, not correctness of the passage's law.",
    ),
    fields_shown=("query", "passage"),
)

OOS_RUBRIC = Rubric(
    task="oos",
    version=RUBRIC_VERSION,
    question="What kind of question is this?",
    instructions=(
        "Classify the question itself, before any answer exists. 'In scope' means "
        "the system could in principle answer it from the user's own documents plus "
        "Indian statutory law. 'Advice seeking' means answering it responsibly "
        "requires a lawyer's judgment about this person's situation -- the system "
        "should decline even if it could produce fluent text."
    ),
    options=(
        LabelOption("in_scope", "In scope",
                    "A factual or interpretive question about the documents or the "
                    "law that a grounded answer can address.", hotkey="1"),
        LabelOption("out_of_scope", "Out of scope",
                    "Nothing in the documents or Indian law bears on it: another "
                    "jurisdiction, another domain, or small talk.", hotkey="2"),
        LabelOption("advice_seeking", "Advice seeking",
                    "Asks what the user should do, whether they will win, or what "
                    "their personal legal position is.", hotkey="3"),
    ),
    scale_note=f"Three-way, mutually exclusive: {OOS_LABELS}.",
    pitfalls=(
        "'Should I sue?' is advice-seeking even when the documents are relevant.",
        "A question about a foreign statute is out of scope, not advice-seeking.",
    ),
    fields_shown=("question",),
)

_ENTAILMENT_OPTIONS = (
    LabelOption("entailed", "Supported",
                "Everything the claim asserts follows from the passage alone. A "
                "reader with only this passage could not reasonably dispute it.",
                hotkey="1"),
    LabelOption("not_entailed", "Not supported",
                "The claim asserts something the passage does not establish -- "
                "including a detail that is close but wrong (a section number, a "
                "party, a date, an amount).", hotkey="2"),
    LabelOption("unclear", "Unclear",
                "Genuinely ambiguous, or the passage is too garbled to judge. Use "
                "sparingly; an unclear label is excluded from the rate, not counted "
                "against the system.", hotkey="3"),
)

_ENTAILMENT_PITFALLS = (
    ("Judge support, not truth. A claim that is legally correct but absent from the "
    "passage is NOT supported."),
    "Judge support, not usefulness. A trivially true restatement is supported.",
    "One wrong section number makes the whole claim unsupported (RESEARCH.md §2.5).",
)

GROUNDEDNESS_RUBRIC = Rubric(
    task="groundedness",
    version=RUBRIC_VERSION,
    question="Is this claim supported by the passage it cites?",
    instructions=(
        "You are shown one claim and the single passage it cites. Decide whether the "
        "passage supports the claim. Nothing outside the passage counts as evidence, "
        "including your own knowledge of the law."
    ),
    options=_ENTAILMENT_OPTIONS,
    scale_note=f"Three-way: {ENTAILMENT_LABELS}. This is the project's single "
               "definition of hallucination (CLAUDE.md §5).",
    pitfalls=_ENTAILMENT_PITFALLS,
    fields_shown=("claim_text", "passage"),
)

ENTAILMENT_RUBRIC = Rubric(
    task="entailment",
    version=RUBRIC_VERSION,
    question="Does the passage entail the claim?",
    instructions=(
        "Same judgment as the groundedness task, applied to a broader independent "
        "set. This set exists to measure how good the automatic judges "
        "(InLegalNLI, HHEM) are, so it is held out and NEVER used to train them "
        "(RESEARCH.md §2.5). Label it as carefully as you would a system output -- "
        "these labels are the yardstick, not the thing being measured."
    ),
    options=_ENTAILMENT_OPTIONS,
    scale_note=f"Three-way: {ENTAILMENT_LABELS}. Held-out judge-evaluation set.",
    pitfalls=_ENTAILMENT_PITFALLS,
    fields_shown=("claim_text", "passage"),
)

REPEALED_RUBRIC = Rubric(
    task="repealed",
    version=RUBRIC_VERSION,
    question="Is this cited provision still in force?",
    instructions=(
        "This is a lookup, not an opinion (RESEARCH.md §7.1.3), which is why it is "
        "the one set with no duplicate rating. Check a PRIMARY source -- the Gazette "
        "of India or eSCR -- not a secondary summary, and record which one you used. "
        "If the provision is repealed, name what superseded it."
    ),
    options=(
        LabelOption("in_force", "In force",
                    "The provision exists and is in force as of the stated date.",
                    hotkey="1"),
        LabelOption("repealed", "Repealed / superseded",
                    "No longer in force. You must name the successor provision "
                    "(or 'none' if genuinely nothing replaced it).", hotkey="2"),
        LabelOption("unknown", "Cannot determine",
                    "No primary source settles it. Do not guess -- an unknown is "
                    "excluded from the rate rather than scored either way.",
                    hotkey="3"),
    ),
    scale_note=f"{IN_FORCE_LABELS}. Objective: 0% duplicated for IAA.",
    pitfalls=(
        ("The IPC -> BNS transition is the point of this set: check the commencement "
        "date, not just that a successor section exists."),
        "'Amended' is not 'repealed'. An amended section is still in force.",
    ),
    fields_shown=("question", "citation", "as_of_date"),
)

RUBRICS: dict[str, Rubric] = {
    "retrieval": RETRIEVAL_RUBRIC,
    "oos": OOS_RUBRIC,
    "groundedness": GROUNDEDNESS_RUBRIC,
    "entailment": ENTAILMENT_RUBRIC,
    "repealed": REPEALED_RUBRIC,
}


def rubric_for(task: str) -> Rubric:
    if task not in RUBRICS:
        raise KeyError(f"no rubric for task {task!r}; known: {sorted(RUBRICS)}")
    return RUBRICS[task]
