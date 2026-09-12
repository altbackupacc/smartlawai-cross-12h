"""Build the SEED gold sets that ship with M6.

    python -m eval.build_seed_gold

Read this before using anything it produces.

**These are seed sets, not the annotation study.** `RESEARCH.md` §7 specifies
300-400 items per set, produced by multiple raters through `eval/annotate/app.py`,
with an inter-annotator agreement statistic. That study needs annotator time,
which M6's code cannot manufacture. What this script produces instead is a
small, honestly-labelled starting set with `n_annotations: 1`, `agreement: null`
and a per-row `provenance` string saying exactly how the label was arrived at.
Nothing here is presented as a human study, and no agreement number is invented.

What that buys, concretely: the harness runs end to end on real data today (the
M6 done-when criterion), the schemas are exercised by real rows rather than
fixtures, and the annotation app has something to load on its first run. Every
row is replaceable -- re-exporting from the app overwrites the file.

Per-set construction, stated plainly:

  groundedness / entailment
      Real IN-Abs (passage, claim) pairs. Positives are headnote sentences whose
      content words are almost entirely present in the matched judgment passage
      (overlap >= SEED_POSITIVE_OVERLAP), so entailment is checkable by reading.
      Negatives are those same pairs with one RESEARCH.md §2.5 perturbation
      applied, which makes them not-entailed BY CONSTRUCTION -- the claim now
      asserts a section number, party, date, court or holding the passage does
      not contain.
      Drawn from IN-Abs **test-data**, while data/perturbations reads
      **train-data**. The two folders share no document ids (verified: 7030 vs
      100 stems, zero overlap), so the disjointness rule RESEARCH.md §2.5 makes
      non-negotiable holds by construction and tests/test_perturbations.py
      enforces it for real rather than skipping.

  oos
      Curated questions. The three-way distinction is largely objective
      (RESEARCH.md §7.1.3 duplicates only 30% of this set for that reason), so a
      single careful labeller is a defensible seed. Still needs the 30%
      duplication pass before it is study data.

  repealed
      Curated IPC->BNS / CrPC->BNSS / Evidence Act->BSA queries. This set is 0%
      duplicated by design because it is an objective registry lookup -- but
      "objective" means "checkable against a primary source", and these seed
      rows are NOT yet checked row-by-row against the Gazette. Every row says so
      in `authority_source`. Verifying them is the first task of whoever runs
      this set for real, and M4's registry is what will do it at scale.

  retrieval
      Queries only, with an EMPTY judgment pool. Not an oversight: `doc_id` and
      `chunk_id` are fresh UUIDs on every ingest (see core/ocr.py and
      core/chunking.py), so committed chunk ids would refer to nothing on the
      next run. Relevance judgments become meaningful when M2 lands a stable
      corpus and stable ids; until then retrieval metrics correctly report
      `unavailable: no_relevance_judgments` rather than a fabricated number.
      The pooling and annotation machinery is ready: ingest, pool with
      eval/annotate/pooling.py, annotate, export.
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval import config
from eval.gold_sets import write_gold_set
from eval.schemas import (
    EntailmentItem,
    GroundednessItem,
    OOSItem,
    RepealedItem,
    RetrievalItem,
)

SEED_RATER = "m6-seed"
SEED_POSITIVE_OVERLAP = 0.85   # near-verbatim, so entailment is checkable by reading
SEED_RUBRIC_VERSION = "v1.0.0"


def _load_generator():
    """Import the perturbation generator by path.

    It is deliberately standalone (no package imports, so it can be released on
    its own -- RESEARCH.md §2.5), which means it is not importable as
    `data.perturbations.generate_perturbations`.
    """
    path = _REPO_ROOT / "data" / "perturbations" / "generate_perturbations.py"
    spec = importlib.util.spec_from_file_location("_m6_perturbations", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the perturbation generator from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves annotations via
    # sys.modules[cls.__module__], which is None for an unregistered module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# oos.jsonl
# --------------------------------------------------------------------------- #
# (question, label, category). Labels follow eval/annotate/guidelines.py's rubric:
# in_scope     -- answerable from the user's documents plus Indian statutory law
# out_of_scope -- nothing in the documents or Indian law bears on it
# advice_seeking -- asks what THIS person should do / will win / is liable for
SEED_OOS: tuple[tuple[str, str, str], ...] = (
    ("What did the court hold about the deductibility of municipal property tax?",
     "in_scope", "holding"),
    ("Which section of the Indian Income Tax Act was in issue in this appeal?",
     "in_scope", "provision"),
    ("Summarise the facts as recorded in the judgment.", "in_scope", "facts"),
    ("What was the ratio decidendi of this decision?", "in_scope", "holding"),
    ("On what date was this appeal decided?", "in_scope", "facts"),
    ("Who appeared as counsel for the appellant?", "in_scope", "facts"),
    ("What is the punishment prescribed for cheating under Indian law?",
     "in_scope", "statute"),
    ("Does this contract contain an indemnity clause?", "in_scope", "clause"),
    ("What is the limitation period stated in the document?", "in_scope", "clause"),
    ("Which High Court's order was under appeal here?", "in_scope", "facts"),
    ("What does the judgment say about the burden of proof?", "in_scope", "holding"),
    (("Explain the difference between a lease and an agreement to lease as set out "
     "in this judgment."), "in_scope", "doctrine"),
    ("What is the capital of France?", "out_of_scope", "general_knowledge"),
    ("Write me a poem about the monsoon.", "out_of_scope", "creative"),
    ("How do I install Python on Windows?", "out_of_scope", "technical"),
    ("What is the current price of gold?", "out_of_scope", "general_knowledge"),
    ("Under Delaware corporate law, what fiduciary duty does a director owe?",
     "out_of_scope", "foreign_jurisdiction"),
    ("What does the UK Theft Act 1968 say about handling stolen goods?",
     "out_of_scope", "foreign_jurisdiction"),
    ("Can you book me a flight to Delhi?", "out_of_scope", "task_request"),
    ("What is the weather in Bengaluru tomorrow?", "out_of_scope", "general_knowledge"),
    ("Translate this judgment into French.", "out_of_scope", "task_request"),
    ("Hello, are you there?", "out_of_scope", "small_talk"),
    ("Should I file this appeal or accept the settlement?",
     "advice_seeking", "strategy"),
    ("Will I win my case if I rely on this judgment?", "advice_seeking", "prediction"),
    ("Am I liable to pay this tax?", "advice_seeking", "personal_liability"),
    ("What should I do about my landlord withholding my deposit?",
     "advice_seeking", "personal_situation"),
    ("Is my employer's termination of my contract legal?",
     "advice_seeking", "personal_liability"),
    ("How much compensation can I claim for my injury?",
     "advice_seeking", "personal_situation"),
    ("Should I plead guilty to this charge?", "advice_seeking", "strategy"),
    ("Do I need a lawyer for this matter, or can I appear myself?",
     "advice_seeking", "strategy"),
    ("Which of my two contracts gives me the stronger position?",
     "advice_seeking", "strategy"),
    ("Can I be arrested for what I posted online yesterday?",
     "advice_seeking", "personal_liability"),
)


# --------------------------------------------------------------------------- #
# repealed.jsonl
# --------------------------------------------------------------------------- #
# (question, citation, in_force_label, superseded_by, category)
#
# The IPC, CrPC and Indian Evidence Act were replaced with effect from
# 1 July 2024 by the Bharatiya Nyaya Sanhita 2023 (Act 45 of 2023), the
# Bharatiya Nagarik Suraksha Sanhita 2023 (Act 46 of 2023) and the Bharatiya
# Sakshya Adhiniyam 2023 (Act 47 of 2023) respectively. Successor sections below
# are seeded from those Acts' own section text and are marked as REQUIRING
# per-row primary-source verification -- see this module's docstring.
BNS_AS_OF = "2026-09-05"
SEED_REPEALED: tuple[tuple[str, str, str, str, str], ...] = (
    ("What is the punishment for murder in India?", "IPC s.302", "repealed",
     "BNS s.103", "offence"),
    ("What is the punishment for cheating?", "IPC s.420", "repealed",
     "BNS s.318(4)", "offence"),
    ("How is culpable homicide not amounting to murder punished?", "IPC s.304",
     "repealed", "BNS s.105", "offence"),
    ("What is the offence of causing death by negligence?", "IPC s.304A",
     "repealed", "BNS s.106(1)", "offence"),
    ("What is the punishment for attempt to murder?", "IPC s.307", "repealed",
     "BNS s.109", "offence"),
    ("How does the law define theft?", "IPC s.378", "repealed", "BNS s.303(1)",
     "definition"),
    ("What is the punishment for theft?", "IPC s.379", "repealed", "BNS s.303(2)",
     "offence"),
    ("What is the punishment for robbery?", "IPC s.392", "repealed", "BNS s.309(4)",
     "offence"),
    ("What does the law say about criminal breach of trust?", "IPC s.405",
     "repealed", "BNS s.316(1)", "definition"),
    ("What is the offence of cruelty by a husband or his relatives?", "IPC s.498A",
     "repealed", "BNS s.85", "offence"),
    ("What is the punishment for criminal conspiracy?", "IPC s.120B", "repealed",
     "BNS s.61(2)", "offence"),
    ("What does 'common intention' mean in Indian criminal law?", "IPC s.34",
     "repealed", "BNS s.3(5)", "doctrine"),
    ("What is the punishment for defamation?", "IPC s.500", "repealed",
     "BNS s.356(2)", "offence"),
    ("How is criminal intimidation punished?", "IPC s.506", "repealed",
     "BNS s.351(2)", "offence"),
    ("What is the punishment for bigamy?", "IPC s.494", "repealed", "BNS s.82(1)",
     "offence"),
    ("What is the punishment for an attempt to commit an offence?", "IPC s.511",
     "repealed", "BNS s.62", "doctrine"),
    ("What was the offence of sedition under Indian law?", "IPC s.124A", "repealed",
     "BNS s.152", "offence"),
    ("What is the punishment for rash driving on a public way?", "IPC s.279",
     "repealed", "BNS s.281", "offence"),
    ("What is the offence of wrongful confinement?", "IPC s.340", "repealed",
     "BNS s.127(1)", "definition"),
    ("Which provision governs the registration of an FIR?", "CrPC s.154", "repealed",
     "BNSS s.173", "procedure"),
    ("Which provision empowers the police to arrest without a warrant?",
     "CrPC s.41", "repealed", "BNSS s.35", "procedure"),
    ("Which provision governs anticipatory bail?", "CrPC s.438", "repealed",
     "BNSS s.482", "procedure"),
    ("Which provision provides for maintenance of wives, children and parents?",
     "CrPC s.125", "repealed", "BNSS s.144", "procedure"),
    ("Which provision deals with the inherent powers of the High Court?",
     "CrPC s.482", "repealed", "BNSS s.528", "procedure"),
    ("What does the law say about the relevancy of confessions to police officers?",
     "Indian Evidence Act s.25", "repealed", "BSA s.23(1)", "evidence"),
    ("Which provision covers the admissibility of electronic records?",
     "Indian Evidence Act s.65B", "repealed", "BSA s.63", "evidence"),
    ("What is the punishment for murder under the Bharatiya Nyaya Sanhita?",
     "BNS s.103", "in_force", "", "offence"),
    ("What is the punishment for cheating under the Bharatiya Nyaya Sanhita?",
     "BNS s.318(4)", "in_force", "", "offence"),
    ("Which provision of the BNSS governs the registration of an FIR?",
     "BNSS s.173", "in_force", "", "procedure"),
    ("Which provision of the Contract Act defines a valid consideration?",
     "Indian Contract Act s.2(d)", "in_force", "", "civil"),
    ("Which provision of the Contract Act voids agreements in restraint of trade?",
     "Indian Contract Act s.27", "in_force", "", "civil"),
    ("Which Article of the Constitution guarantees equality before the law?",
     "Constitution of India art.14", "in_force", "", "constitutional"),
    ("Which Article protects against self-incrimination?",
     "Constitution of India art.20(3)", "in_force", "", "constitutional"),
    ("Which provision of the Limitation Act governs suits for recovery of money?",
     "Limitation Act 1963 art.19", "in_force", "", "civil"),
)


# --------------------------------------------------------------------------- #
# retrieval.jsonl -- queries only (see module docstring)
# --------------------------------------------------------------------------- #
SEED_RETRIEVAL_QUERIES: tuple[tuple[str, str], ...] = (
    ("What deduction did the assessee claim and on what basis?", "deduction"),
    ("What did the High Court hold before the appeal?", "procedural_history"),
    ("Which statutory provision was construed by the Court?", "provision"),
    ("What is the meaning the Court gave to 'capital charge'?", "definition"),
    ("On what ground was the appeal allowed or dismissed?", "outcome"),
    ("What were the facts giving rise to the dispute?", "facts"),
    ("Which earlier decisions did the Court rely on?", "precedent"),
    ("What order as to costs did the Court make?", "outcome"),
    ("What was the question referred to the High Court?", "reference"),
    ("How did the Court distinguish the authorities cited by the appellant?",
     "reasoning"),
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_oos() -> list[OOSItem]:
    return [
        OOSItem(
            item_id=f"oos-seed-{i:04d}", question=question, label=label,
            category=category, rater_ids=[SEED_RATER], n_annotations=1,
            agreement=None, rubric_version=SEED_RUBRIC_VERSION,
            provenance="seed: curated by the M6 implementer against "
                       "eval/annotate/guidelines.py's OOS rubric; single labeller, "
                       "no IAA pass yet (RESEARCH.md §7.1.3 requires 30%)")
        for i, (question, label, category) in enumerate(SEED_OOS)
    ]


def build_repealed() -> list[RepealedItem]:
    items = []
    for i, (question, citation, label, successor, category) in enumerate(SEED_REPEALED):
        items.append(RepealedItem(
            item_id=f"repealed-seed-{i:04d}", question=question, citation=citation,
            in_force_label=label, superseded_by=successor, as_of_date=BNS_AS_OF,
            category=category, rater_ids=[SEED_RATER], n_annotations=1,
            agreement=None, rubric_version=SEED_RUBRIC_VERSION,
            authority_source="BNS 2023 (Act 45) / BNSS 2023 (Act 46) / BSA 2023 "
                             "(Act 47), in force 2024-07-01 -- SEED VALUE, NOT YET "
                             "VERIFIED ROW-BY-ROW AGAINST THE GAZETTE",
            provenance="seed: successor mapping taken from the replacing Act's "
                       "section text. RESEARCH.md §7.1.3 calls this set objective, "
                       "which makes primary-source verification mandatory before "
                       "any reported number -- that verification has NOT been done."))
    return items


def build_retrieval() -> list[RetrievalItem]:
    return [
        RetrievalItem(
            item_id=f"retrieval-seed-{i:04d}", query=query, category=category,
            judgments={}, pooled_from=[], rater_ids=[], n_annotations=0,
            agreement=None,
            provenance="seed: query only. Judgments are empty because doc_id and "
                       "chunk_id are fresh UUIDs on every ingest, so committed "
                       "chunk ids would refer to nothing. Pool and judge once M2 "
                       "provides a stable corpus (eval/annotate/pooling.py).")
        for i, (query, category) in enumerate(SEED_RETRIEVAL_QUERIES)
    ]


def _claim_passage_pairs(n: int, seed: int, gen) -> list[tuple]:
    """Real IN-Abs test-data pairs: (doc_id, passage, positive_claim,
    perturbed_claim, ptype, detail, overlap).

    Per I5 and test_data_leakage.py, only documents in data/splits/test.json
    (and never train.json) are used for evaluation gold sets.
    """
    import json
    splits_test_path = _REPO_ROOT / "data" / "splits" / "test.json"
    splits_train_path = _REPO_ROOT / "data" / "splits" / "train.json"

    test_ids = None
    if splits_test_path.exists():
        test_ids = set(json.loads(splits_test_path.read_text(encoding="utf-8")).get("doc_ids", []))
    train_ids = set()
    if splits_train_path.exists():
        train_ids = set(json.loads(splits_train_path.read_text(encoding="utf-8")).get("doc_ids", []))

    rng = random.Random(seed)
    out = []
    for doc_id, judgment, summary in gen.iter_documents(config.IN_ABS_DIR,
                                                        "test-data"):
        if test_ids and doc_id not in test_ids:
            continue
        if doc_id in train_ids:
            continue
        if len(out) >= n:
            break
        for pair in gen.extract_pairs(doc_id, judgment, summary,
                                      min_overlap=SEED_POSITIVE_OVERLAP,
                                      max_pairs_per_doc=2):
            if len(out) >= n:
                break
            applicable = []
            for ptype in gen.PERTURBATION_TYPES:
                result = gen.PERTURBERS[ptype](pair.claim, rng)
                if result is not None and result[0] != pair.claim:
                    applicable.append((ptype, result[0], result[1]))
            if not applicable:
                continue
            ptype, perturbed, detail = applicable[0]
            out.append((doc_id, pair.passage, pair.claim, perturbed, ptype,
                        detail, pair.overlap))
    return out


def build_claim_sets(n_each: int, seed: int) -> tuple[list, list]:
    """groundedness.jsonl and entailment.jsonl from disjoint IN-Abs test docs."""
    gen = _load_generator()
    pairs = _claim_passage_pairs(n_each * 2, seed, gen)
    half = len(pairs) // 2
    groundedness_pairs, entailment_pairs = pairs[:half], pairs[half:]

    def rows(source, cls, prefix, extra_note):
        items = []
        for i, (doc_id, passage, positive, perturbed, ptype, detail,
                overlap) in enumerate(source):
            common = {"source_doc_id": doc_id, "passage": passage,
                          "rater_ids": [SEED_RATER], "n_annotations": 1, "agreement": None,
                          "rubric_version": SEED_RUBRIC_VERSION,
                          "split": "", "category": "in_abs_test"}
            items.append(cls(
                item_id=f"{prefix}-seed-{i:04d}-pos", claim_text=positive,
                label="entailed", notes=f"content-word overlap {overlap}",
                provenance=f"seed: IN-Abs headnote sentence with >= "
                           f"{SEED_POSITIVE_OVERLAP} content-word overlap with its "
                           f"matched judgment passage, so entailment is checkable "
                           f"by reading. {extra_note}", **common))
            items.append(cls(
                item_id=f"{prefix}-seed-{i:04d}-neg", claim_text=perturbed,
                label="not_entailed",
                notes=f"perturbation={ptype} {detail}",
                provenance=f"seed: the positive above with one RESEARCH.md §2.5 "
                           f"perturbation ({ptype}) applied, which makes it "
                           f"not-entailed by construction. {extra_note}", **common))
        return items

    groundedness = rows(groundedness_pairs, GroundednessItem, "grounded",
                        "Single labeller, no IAA pass yet (100% duplication "
                        "required by RESEARCH.md §7.1.3).")
    entailment = [
        EntailmentItem(**{**item.to_row(), "item_id": item.item_id.replace(
            "grounded", "entail"), "judge_source": "internal"})
        for item in rows(entailment_pairs, GroundednessItem, "grounded",
                         "HELD OUT: never used to train InLegalNLI "
                         "(RESEARCH.md §2.5).")
    ]
    return groundedness, entailment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-dir", type=Path, default=None)
    parser.add_argument("--n-claim-pairs", type=int, default=20,
                        help="source pairs per claim set (each yields 2 rows)")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = parser.parse_args(argv)

    written: dict[str, int] = {}
    for name, items in (("oos", build_oos()),
                        ("repealed", build_repealed()),
                        ("retrieval", build_retrieval())):
        write_gold_set(name, items, args.gold_dir)
        written[name] = len(items)

    groundedness, entailment = build_claim_sets(args.n_claim_pairs, args.seed)
    write_gold_set("groundedness", groundedness, args.gold_dir)
    write_gold_set("entailment", entailment, args.gold_dir)
    written["groundedness"] = len(groundedness)
    written["entailment"] = len(entailment)

    for name, count in sorted(written.items()):
        print(f"{name}: {count} seed items")
    print("\nThese are SEED sets with n_annotations=1 and no agreement statistic. "
          "The RESEARCH.md §7 study replaces them via eval/annotate/app.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
