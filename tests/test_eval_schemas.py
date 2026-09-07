"""Gold-set schema validation (M6 Phase 2).

Validation has to be real, not decorative: these schemas are the only thing
standing between a hand-edited JSONL row and a number in the paper.
"""
from __future__ import annotations

import pytest

from eval import config
from eval.schemas import (
    ENTAILMENT_LABELS,
    GOLD_SET_NAMES,
    GOLD_SET_SPECS,
    IN_FORCE_LABELS,
    OOS_LABELS,
    Annotation,
    EntailmentItem,
    GroundednessItem,
    OOSItem,
    RepealedItem,
    RetrievalItem,
    SchemaError,
    validate_row,
)


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #
def test_oos_round_trip():
    item = OOSItem(item_id="oos-1", question="What is s.302?", label="in_scope")
    assert OOSItem.from_row(item.to_row()) == item


def test_retrieval_round_trip_preserves_graded_judgments():
    item = RetrievalItem(item_id="r-1", query="q", judgments={"c1": 2, "c2": 0})
    assert RetrievalItem.from_row(item.to_row()).judgments == {"c1": 2, "c2": 0}


def test_groundedness_round_trip():
    item = GroundednessItem(item_id="g-1", claim_text="c", passage="p",
                            label="entailed", passage_ids=["ch-1"])
    assert GroundednessItem.from_row(item.to_row()).passage_ids == ["ch-1"]


def test_repealed_round_trip():
    item = RepealedItem(item_id="rp-1", question="q", citation="IPC s.302",
                        in_force_label="repealed", superseded_by="BNS s.103")
    assert RepealedItem.from_row(item.to_row()).superseded_by == "BNS s.103"


# --------------------------------------------------------------------------- #
# Required fields and label domains
# --------------------------------------------------------------------------- #
def test_missing_required_field_is_rejected():
    with pytest.raises(SchemaError, match="question"):
        OOSItem(item_id="oos-1", question="", label="in_scope").validate()


def test_blank_item_id_is_rejected():
    with pytest.raises(SchemaError, match="item_id"):
        OOSItem(item_id="   ", question="q", label="in_scope").validate()


def test_unknown_field_is_rejected_rather_than_silently_dropped():
    with pytest.raises(SchemaError, match="unknown field"):
        OOSItem.from_row({"item_id": "x", "question": "q", "label": "in_scope",
                          "labl": "typo"})


def test_oos_label_outside_domain_is_rejected():
    with pytest.raises(SchemaError, match="not in"):
        OOSItem(item_id="x", question="q", label="maybe").validate()


def test_entailment_label_outside_domain_is_rejected():
    with pytest.raises(SchemaError, match="not in"):
        GroundednessItem(item_id="x", claim_text="c", passage="p",
                         label="probably").validate()


def test_relevance_grade_outside_domain_is_rejected():
    with pytest.raises(SchemaError, match="must be one of"):
        RetrievalItem(item_id="x", query="q", judgments={"c1": 7}).validate()


def test_repealed_row_must_name_its_successor():
    """A repealed provision with no successor is an incomplete judgment, and
    the IPC->BNS experiment depends on that field."""
    with pytest.raises(SchemaError, match="superseded"):
        RepealedItem(item_id="x", question="q", citation="IPC s.302",
                     in_force_label="repealed", superseded_by="").validate()


def test_repealed_row_in_force_needs_no_successor():
    RepealedItem(item_id="x", question="q", citation="BNS s.103",
                 in_force_label="in_force").validate()


# --------------------------------------------------------------------------- #
# Semantics the metrics depend on
# --------------------------------------------------------------------------- #
def test_unclear_label_maps_to_none_not_false():
    """I2: an unclear human judgment is not evidence of a hallucination."""
    assert GroundednessItem(item_id="x", claim_text="c", passage="p",
                            label="unclear").supported() is None
    assert GroundednessItem(item_id="x", claim_text="c", passage="p",
                            label="entailed").supported() is True
    assert GroundednessItem(item_id="x", claim_text="c", passage="p",
                            label="not_entailed").supported() is False


def test_oos_should_refuse_covers_advice_seeking():
    assert OOSItem(item_id="x", question="q", label="advice_seeking").should_refuse()
    assert OOSItem(item_id="x", question="q", label="out_of_scope").should_refuse()
    assert not OOSItem(item_id="x", question="q", label="in_scope").should_refuse()


def test_retrieval_grade_zero_is_judged_irrelevant_not_unjudged():
    item = RetrievalItem(item_id="x", query="q", judgments={"c1": 0, "c2": 2})
    assert item.relevant_ids() == {"c2"}
    assert item.gains() == {"c2": 2.0}       # grade 0 stays out of the ideal ranking
    assert "c1" in item.judgments             # but it IS judged


def test_repealed_item_presents_m4s_own_registry_result_shape():
    """M4_ONBOARDING.md §6's RegistryResult contract is {status, as_of, note} --
    reused verbatim (audit-corrected, docs/M6_GAP_ANALYSIS.md G2) rather than an
    invented shape, so a real M4 result needs no translation layer."""
    r = RepealedItem(item_id="x", question="q", citation="IPC s.420",
                     in_force_label="repealed",
                     superseded_by="BNS s.318(4)").as_registry_result()
    assert r["status"] == "repealed"
    assert "BNS s.318(4)" in r["note"]


def test_in_force_item_carries_no_note():
    r = RepealedItem(item_id="x", question="q", citation="BNS s.318(4)",
                     in_force_label="in_force").as_registry_result()
    assert r["status"] == "in_force" and r["note"] is None


def test_unknown_in_force_label_maps_to_not_found_rather_than_in_force():
    """Defaulting 'unknown' to in-force would understate the repealed rate.
    'not_found' is M4's own honest 'I don't know' status (M4_ONBOARDING §6)."""
    r = RepealedItem(item_id="x", question="q", citation="Obscure s.1",
                     in_force_label="unknown").as_registry_result()
    assert r["status"] == "not_found"


# --------------------------------------------------------------------------- #
# Annotation record
# --------------------------------------------------------------------------- #
def _annotation(**overrides):
    row = {"annotation_id": "a-1", "task": "oos", "item_id": "i-1", "rater_id": "r1",
               "label": "in_scope", "mode": "correction",
               "created_at": "2026-01-01T00:00:00+00:00", "seconds_on_item": 4.2}
    row.update(overrides)
    return Annotation(**row)


def test_annotation_round_trip():
    assert Annotation.from_row(_annotation().to_row()).rater_id == "r1"


def test_annotation_rejects_unknown_mode():
    with pytest.raises(SchemaError, match="mode"):
        _annotation(mode="guessing").validate()


def test_annotation_rejects_negative_time():
    with pytest.raises(SchemaError, match="seconds_on_item"):
        _annotation(seconds_on_item=-1).validate()


def test_cold_mode_may_not_carry_a_model_proposal():
    """Showing a proposal in cold mode would defeat the anchoring control."""
    with pytest.raises(SchemaError, match="anchoring"):
        _annotation(mode="cold", proposed_label="in_scope").validate()


def test_annotation_requires_rater_and_item():
    with pytest.raises(SchemaError):
        _annotation(rater_id="").validate()


# --------------------------------------------------------------------------- #
# The gold-set registry
# --------------------------------------------------------------------------- #
def test_all_five_gold_sets_are_registered():
    assert set(GOLD_SET_NAMES) == {"retrieval", "oos", "groundedness", "repealed",
                                   "entailment"}


def test_iaa_duplication_rates_match_the_research_protocol():
    """RESEARCH.md §7.1.3: 100% where subjectivity lives, 30% for OOS, 0% for
    repealed (objective registry lookup)."""
    rates = {n: s.iaa_duplication_rate for n, s in GOLD_SET_SPECS.items()}
    assert rates == {"retrieval": 1.00, "groundedness": 1.00, "entailment": 1.00,
                     "oos": 0.30, "repealed": 0.00}
    assert rates == config.IAA_DUPLICATION_RATE


def test_entailment_is_flagged_never_train_on_and_groundedness_is_not():
    """RESEARCH.md §2.5 hard rule -- the two sets must stay distinguishable."""
    assert GOLD_SET_SPECS["entailment"].never_train_on is True
    assert GOLD_SET_SPECS["groundedness"].never_train_on is False
    assert GOLD_SET_SPECS["entailment"].schema is EntailmentItem
    assert GOLD_SET_SPECS["groundedness"].schema is GroundednessItem
    assert EntailmentItem is not GroundednessItem


def test_label_domains_are_wired_to_the_specs():
    assert GOLD_SET_SPECS["oos"].labels == OOS_LABELS
    assert GOLD_SET_SPECS["groundedness"].labels == ENTAILMENT_LABELS
    assert GOLD_SET_SPECS["repealed"].labels == IN_FORCE_LABELS


def test_validate_row_dispatches_by_task():
    item = validate_row("oos", {"item_id": "x", "question": "q",
                                "label": "in_scope"})
    assert isinstance(item, OOSItem)


def test_validate_row_rejects_unknown_task():
    with pytest.raises(SchemaError, match="unknown gold set"):
        validate_row("nonexistent", {"item_id": "x"})
