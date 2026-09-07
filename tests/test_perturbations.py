"""Perturbation generator tests (M6 Phase 9) -- RESEARCH.md §2.5.

The disjointness checks here are the reason the generator lives under M6 at all
(M6_ONBOARDING.md §9). Two of them exist:

  * **Disjoint from every evaluation gold set** -- enforced NOW, as a real
    assertion, because the seed gold sets are built from IN-Abs test-data while
    the generator reads train-data, and the two folders share no document ids.
  * **Drawn from the TRAINING split specifically** -- `pytest.skip`-guarded with
    a stated reason until `data/splits/*.json` exists, then a real assertion with
    no further code changes. Same convention M1_ONBOARDING.md §9 established for
    its own leakage test; an honest skip, not a vacuous pass.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from eval import config
from eval.gold_sets import load_gold_set
from eval.schemas import GOLD_SET_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_PATH = REPO_ROOT / "data" / "perturbations" / "generate_perturbations.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("_test_perturbations",
                                                  GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # @dataclass needs it registered
    spec.loader.exec_module(module)
    return module


gen = _load_generator()

CORPUS_AVAILABLE = (config.IN_ABS_DIR / "train-data" / "judgement").is_dir()
requires_corpus = pytest.mark.skipif(
    not CORPUS_AVAILABLE,
    reason=f"IN-Abs corpus not present at {config.IN_ABS_DIR} (M1 owns data/raw/)")


# --------------------------------------------------------------------------- #
# The seven perturbation types (RESEARCH.md §2.5's taxonomy)
# --------------------------------------------------------------------------- #
def test_all_seven_taxonomy_types_are_implemented():
    assert set(gen.PERTURBATION_TYPES) == set(gen.PERTURBERS)
    assert set(gen.PERTURBATION_TYPES) == set(config.PERTURBATION_TYPES)
    assert len(gen.PERTURBATION_TYPES) == 7


def _apply(ptype, claim, seed=42):
    return gen.PERTURBERS[ptype](claim, random.Random(seed))


def test_section_number_perturbation_changes_the_provision():
    claim = "The Court construed section 9 of the Indian Income Tax Act."
    perturbed, detail = _apply("section_number", claim)
    assert perturbed != claim
    assert "section 9 " not in perturbed
    assert detail["from"] and detail["to"]


def test_ipc_bns_swap_renames_the_statute_without_renumbering():
    """The exact error a model trained on pre-2024 text makes."""
    claim = "The accused was convicted under section 420 of the Indian Penal Code."
    perturbed, detail = _apply("ipc_bns_swap", claim)
    assert "420" in perturbed                       # number deliberately unchanged
    assert "Indian Penal Code" not in perturbed
    assert detail["to"] in ("Bharatiya Nyaya Sanhita", "BNS")


def test_negation_reverses_legal_effect():
    perturbed, _ = _apply("negate_obligation", "The tenant shall pay the rent.")
    assert "shall not pay" in perturbed


def test_negation_removes_an_existing_negation_rather_than_doubling_it():
    perturbed, _ = _apply("negate_obligation", "The tenant shall not pay the rent.")
    assert "shall not not" not in perturbed
    assert "shall pay" in perturbed


def test_party_swap_reverses_who_won():
    perturbed, detail = _apply("swap_parties",
                               "The appellant succeeded against the respondent.")
    assert perturbed.startswith("The respondent")
    assert "against the appellant" in perturbed
    assert detail["n_swapped"] == 2


def test_party_swap_preserves_case():
    perturbed, _ = _apply("swap_parties", "Appellant and respondent both appealed.")
    assert perturbed.startswith("Respondent")


def test_date_and_amount_perturbations_change_the_fact():
    perturbed, detail = _apply("alter_date_amount",
                               "The Bombay Finance Act, 1939 created the charge.")
    assert "1939" not in perturbed and detail["kind"] == "year"

    perturbed, detail = _apply("alter_date_amount", "He was awarded Rs. 5,000.")
    assert "5,000" not in perturbed and detail["kind"] == "amount"


def test_court_change_alters_precedential_weight():
    perturbed, detail = _apply("change_court", "The Supreme Court allowed the appeal.")
    assert "Supreme Court" not in perturbed
    assert detail["from"] == "Supreme Court"


def test_court_change_prefers_the_longest_match():
    """'Bombay High Court' must not be clobbered by the substring 'High Court'."""
    _, detail = _apply("change_court", "The Bombay High Court so held.")
    assert detail["from"] == "Bombay High Court"


def test_unsupported_holding_appends_a_fabricated_ratio():
    claim = "The appeal was allowed."
    perturbed, detail = _apply("unsupported_holding", claim)
    assert perturbed.startswith(claim)
    assert len(perturbed) > len(claim)
    assert detail["inserted"] in gen._FABRICATED_HOLDINGS


def test_inapplicable_perturbation_returns_none_rather_than_faking_one():
    """A type that cannot fire is skipped, so per-type detection rates measure
    real instances only."""
    plain = "The weather that day was pleasant."
    for ptype in ("section_number", "ipc_bns_swap", "swap_parties", "change_court"):
        assert _apply(ptype, plain) is None


def test_every_perturbation_actually_changes_the_claim():
    claim = ("On 12th May 1975 the Supreme Court held that the appellant shall pay "
             "Rs. 5,000 to the respondent under section 420 of the Indian Penal Code.")
    for ptype in gen.PERTURBATION_TYPES:
        result = _apply(ptype, claim)
        assert result is not None, ptype
        assert result[0] != claim, ptype


# --------------------------------------------------------------------------- #
# Source-pair extraction
# --------------------------------------------------------------------------- #
def test_overlap_score_is_a_fraction_of_claim_content_words():
    assert gen.overlap_score("the appellant paid rent", "appellant paid rent") == 1.0
    assert gen.overlap_score("zebra giraffe elephant", "appellant paid rent") == 0.0


def test_overlap_of_an_empty_claim_is_zero():
    assert gen.overlap_score("", "some passage") == 0.0


def test_extract_pairs_respects_the_overlap_threshold():
    judgment = "The appellant was granted an allotment of lands in Raikot in 1949."
    summary = "Completely unrelated sentence about maritime insurance contracts here."
    assert gen.extract_pairs("d1", judgment, summary, min_overlap=0.9) == []


def test_extract_pairs_returns_matched_windows():
    judgment = ("Preamble text here. The appellant was granted an allotment of "
                "lands in Raikot in 1949. Further discussion follows below.")
    summary = "The appellant was granted an allotment of lands in Raikot in 1949."
    pairs = gen.extract_pairs("d1", judgment, summary, min_overlap=0.5)
    assert pairs and pairs[0].doc_id == "d1"
    assert "Raikot" in pairs[0].passage


# --------------------------------------------------------------------------- #
# Generation, determinism, tagging
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    if not CORPUS_AVAILABLE:
        pytest.skip("IN-Abs corpus not present")
    out = tmp_path_factory.mktemp("perturbations") / "perturbations.jsonl"
    manifest = gen.generate(out_path=out, n=40, seed=42)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return manifest, rows, out


@requires_corpus
def test_generator_produces_rows(generated):
    _, rows, _ = generated
    assert rows


@requires_corpus
def test_every_row_is_tagged_with_its_source_doc_id(generated):
    """Non-negotiable: disjointness cannot be checked without it."""
    _, rows, _ = generated
    for row in rows:
        assert row["source_doc_id"]
        assert row["perturbation_type"] in set(gen.PERTURBATION_TYPES) | {"none"}
        assert row["label"] in ("entailed", "not_entailed")
        assert row["seed"] == 42
        assert row["generator_version"] == gen.GENERATOR_VERSION


@requires_corpus
def test_perturbed_rows_actually_differ_from_their_source_claim(generated):
    _, rows, _ = generated
    negatives = [r for r in rows if r["label"] == "not_entailed"]
    assert negatives
    for row in negatives:
        assert row["claim"] != row["original_claim"]


@requires_corpus
def test_positive_rows_are_the_unmodified_source_claim(generated):
    _, rows, _ = generated
    for row in [r for r in rows if r["label"] == "entailed"]:
        assert row["claim"] == row["original_claim"]
        assert row["perturbation_type"] == "none"


@requires_corpus
def test_generation_is_deterministic_under_a_fixed_seed(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    gen.generate(out_path=first, n=20, seed=42)
    gen.generate(out_path=second, n=20, seed=42)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


@requires_corpus
def test_a_different_seed_produces_different_output(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    gen.generate(out_path=first, n=20, seed=42)
    gen.generate(out_path=second, n=20, seed=1337)
    assert first.read_text(encoding="utf-8") != second.read_text(encoding="utf-8")


@requires_corpus
def test_type_assignment_is_balanced_rather_than_dominated_by_one_type(generated):
    """`unsupported_holding` always applies; without balancing it would swamp
    the set and make per-type detection rates incomparable."""
    manifest, _, _ = generated
    counts = manifest["type_counts"]
    fired = {t: c for t, c in counts.items() if c > 0}
    assert len(fired) >= 3
    assert counts["unsupported_holding"] <= sum(counts.values()) * 0.6


@requires_corpus
def test_manifest_records_everything_needed_to_reproduce_the_run(generated):
    manifest, rows, _ = generated
    assert manifest["seed"] == 42
    assert manifest["n_rows"] == len(rows)
    assert manifest["generator_version"] == gen.GENERATOR_VERSION
    assert manifest["min_overlap"] == gen.DEFAULT_MIN_OVERLAP
    assert "frozen_splits_available" in manifest


@requires_corpus
def test_manifest_is_written_next_to_the_output(generated):
    _, _, out = generated
    assert out.with_suffix(".manifest.json").exists()


@requires_corpus
def test_positives_can_be_suppressed(tmp_path):
    out = tmp_path / "neg_only.jsonl"
    gen.generate(out_path=out, n=20, seed=42, include_positives=False)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows and all(r["label"] == "not_entailed" for r in rows)


def test_generator_is_releasable_standalone():
    """RESEARCH.md §2.5 requires publishing the generator itself, so it must not
    import this repo's packages."""
    source = GENERATOR_PATH.read_text(encoding="utf-8")
    for forbidden in ("from eval", "import eval", "from smartlawai",
                      "import smartlawai", "from data.", "import datasets"):
        assert forbidden not in source, f"generator imports {forbidden!r}"


# --------------------------------------------------------------------------- #
# Split safety -- the reason this task lives under M6
# --------------------------------------------------------------------------- #
@requires_corpus
def test_perturbations_are_disjoint_from_every_evaluation_gold_set(generated):
    """RESEARCH.md §2.5, non-negotiable. Enforced now, not skipped: the
    generator reads IN-Abs train-data and the gold sets are built from
    test-data, and the two folders share no document ids."""
    _, rows, _ = generated
    perturbation_docs = {row["source_doc_id"] for row in rows}

    for name in GOLD_SET_NAMES:
        gold_docs = {item.source_doc_id for item in load_gold_set(name)
                     if item.source_doc_id}
        overlap = perturbation_docs & gold_docs
        assert not overlap, (
            f"{len(overlap)} document(s) appear in both the perturbation training "
            f"set and eval/gold/{name}.jsonl: {sorted(overlap)[:5]}")


@requires_corpus
def test_perturbations_never_come_from_the_held_out_entailment_documents(generated):
    """The judge-evaluation set must stay independent of the judge's training
    data or its numbers mean nothing (RESEARCH.md §2.5 hard rules)."""
    _, rows, _ = generated
    entailment_docs = {i.source_doc_id for i in load_gold_set("entailment")
                       if i.source_doc_id}
    assert entailment_docs, "entailment.jsonl carries no source_doc_id to check"
    assert not {r["source_doc_id"] for r in rows} & entailment_docs


@requires_corpus
def test_perturbation_sources_are_drawn_from_the_training_split(generated):
    """Perturbations built from dev/test documents would leak model familiarity
    with those documents into a judge later evaluated on them.

    Skipped with a stated reason until M1 freezes data/splits/*.json; becomes a
    real assertion at that moment with no code change here (M1_ONBOARDING.md §9's
    convention)."""
    train_ids = gen.load_split_doc_ids(config.SPLITS_DIR, "train")
    if train_ids is None:
        pytest.skip(f"no {config.SPLITS_DIR / 'train.json'} yet -- M1 has not "
                    f"frozen the document-level splits (I5), so the "
                    f"train-split-only restriction cannot be enforced")
    _, rows, _ = generated
    offenders = {r["source_doc_id"] for r in rows} - train_ids
    assert not offenders, (
        f"{len(offenders)} perturbation source document(s) are not in the training "
        f"split: {sorted(offenders)[:5]}")


def test_split_loader_returns_none_when_splits_do_not_exist(tmp_path):
    """None means 'not frozen yet', which the caller must handle explicitly --
    it must never be confused with an empty split."""
    assert gen.load_split_doc_ids(tmp_path, "train") is None


def test_split_loader_reads_a_list_or_a_keyed_dict(tmp_path):
    (tmp_path / "train.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert gen.load_split_doc_ids(tmp_path, "train") == {"a", "b"}
    (tmp_path / "dev.json").write_text(json.dumps({"doc_ids": ["c"]}),
                                       encoding="utf-8")
    assert gen.load_split_doc_ids(tmp_path, "dev") == {"c"}


@requires_corpus
def test_manifest_states_plainly_when_split_safety_could_not_be_enforced(generated):
    manifest, _, _ = generated
    if not manifest["frozen_splits_available"]:
        assert "split_safety_note" in manifest
        assert "could NOT be enforced" in manifest["split_safety_note"]
    else:
        assert manifest["train_split_filter_applied"] is True
