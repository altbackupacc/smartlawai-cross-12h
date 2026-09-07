"""Gold-set loading, validation, freezing and the power calculation (M6 Phase 3)."""
from __future__ import annotations

import json

import pytest

from eval import config
from eval.gold_sets import (
    GoldSetMissing,
    describe_gold_set,
    gold_path,
    load_all_gold_sets,
    load_gold_set,
    proportion_ci_halfwidth,
    sizing_note,
    write_gold_set,
)
from eval.schemas import GOLD_SET_NAMES, GOLD_SET_SPECS, OOSItem, SchemaError


# --------------------------------------------------------------------------- #
# The sets actually shipped in the repo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", GOLD_SET_NAMES)
def test_every_shipped_gold_set_loads_and_validates(name):
    items = load_gold_set(name)
    assert items, f"{name}.jsonl is empty"
    for item in items:
        item.validate()


@pytest.mark.parametrize("name", GOLD_SET_NAMES)
def test_every_shipped_item_has_a_unique_id(name):
    ids = [i.item_id for i in load_gold_set(name)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name", GOLD_SET_NAMES)
def test_every_shipped_item_states_its_provenance(name):
    """A gold row with no provenance cannot be defended in a paper."""
    for item in load_gold_set(name):
        assert item.provenance.strip(), f"{name}:{item.item_id} has no provenance"


@pytest.mark.parametrize("name", GOLD_SET_NAMES)
def test_seed_sets_do_not_claim_an_agreement_they_do_not_have(name):
    """No invented kappa. A single-rater row reports agreement=None."""
    for item in load_gold_set(name):
        if item.n_annotations < 2:
            assert item.agreement is None, (
                f"{name}:{item.item_id} claims agreement with "
                f"{item.n_annotations} annotation(s)")


def test_load_all_reports_every_set_and_nothing_missing():
    loaded, missing = load_all_gold_sets()
    assert set(loaded) == set(GOLD_SET_NAMES)
    assert missing == {}


def test_groundedness_and_entailment_are_disjoint_files():
    """RESEARCH.md §2.5: they must never be merged by accident."""
    g_ids = {i.item_id for i in load_gold_set("groundedness")}
    e_ids = {i.item_id for i in load_gold_set("entailment")}
    assert not (g_ids & e_ids)


def test_repealed_set_contains_both_classes():
    """An all-repealed set would make the IPC->BNS rate uninformative."""
    labels = {i.in_force_label for i in load_gold_set("repealed")}
    assert {"in_force", "repealed"} <= labels


def test_oos_set_covers_all_three_classes():
    labels = {i.label for i in load_gold_set("oos")}
    assert labels == {"in_scope", "out_of_scope", "advice_seeking"}


# --------------------------------------------------------------------------- #
# IO behaviour
# --------------------------------------------------------------------------- #
def test_missing_gold_set_raises_rather_than_returning_empty(tmp_path):
    """An empty list becomes a measured 0.0 downstream. It must raise."""
    with pytest.raises(GoldSetMissing, match="not found"):
        load_gold_set("oos", tmp_path)


def test_load_all_records_missing_sets_with_a_reason(tmp_path):
    loaded, missing = load_all_gold_sets(tmp_path)
    assert loaded == {}
    assert set(missing) == set(GOLD_SET_NAMES)
    assert all("not found" in reason for reason in missing.values())


def test_malformed_json_names_its_line_number(tmp_path):
    path = tmp_path / GOLD_SET_SPECS["oos"].filename
    path.write_text('{"item_id": "a", "question": "q", "label": "in_scope"}\n'
                    "{not json}\n", encoding="utf-8")
    with pytest.raises(SchemaError, match=r":2:"):
        load_gold_set("oos", tmp_path)


def test_invalid_row_names_its_line_number(tmp_path):
    path = tmp_path / GOLD_SET_SPECS["oos"].filename
    path.write_text('{"item_id": "a", "question": "q", "label": "in_scope"}\n'
                    '{"item_id": "b", "question": "q", "label": "nope"}\n',
                    encoding="utf-8")
    with pytest.raises(SchemaError, match=r":2:"):
        load_gold_set("oos", tmp_path)


def test_blank_lines_and_comments_are_skipped(tmp_path):
    path = tmp_path / GOLD_SET_SPECS["oos"].filename
    path.write_text("// a note\n\n"
                    '{"item_id": "a", "question": "q", "label": "in_scope"}\n',
                    encoding="utf-8")
    assert len(load_gold_set("oos", tmp_path)) == 1


def test_write_then_load_round_trips(tmp_path):
    items = [OOSItem(item_id="a", question="q1", label="in_scope"),
             OOSItem(item_id="b", question="q2", label="advice_seeking")]
    write_gold_set("oos", items, tmp_path)
    assert [i.item_id for i in load_gold_set("oos", tmp_path)] == ["a", "b"]


def test_write_validates_before_writing(tmp_path):
    with pytest.raises(SchemaError):
        write_gold_set("oos", [OOSItem(item_id="a", question="q", label="bad")],
                       tmp_path)


def test_write_is_deterministic_so_reexports_produce_reviewable_diffs(tmp_path):
    items = [OOSItem(item_id="a", question="q", label="in_scope")]
    write_gold_set("oos", items, tmp_path)
    first = (tmp_path / "oos.jsonl").read_text(encoding="utf-8")
    write_gold_set("oos", items, tmp_path)
    assert (tmp_path / "oos.jsonl").read_text(encoding="utf-8") == first


def test_gold_path_rejects_unknown_set():
    with pytest.raises(SchemaError, match="unknown gold set"):
        gold_path("nope")


# --------------------------------------------------------------------------- #
# Description, checksums, power calculation
# --------------------------------------------------------------------------- #
def test_describe_records_a_checksum_so_silent_edits_are_detectable(tmp_path):
    items = [OOSItem(item_id="a", question="q", label="in_scope")]
    write_gold_set("oos", items, tmp_path)
    before = describe_gold_set("oos", items, tmp_path).sha256
    assert len(before) == 64

    items.append(OOSItem(item_id="b", question="q2", label="out_of_scope"))
    write_gold_set("oos", items, tmp_path)
    assert describe_gold_set("oos", items, tmp_path).sha256 != before


def test_describe_counts_labels_and_duplication(tmp_path):
    items = [OOSItem(item_id="a", question="q", label="in_scope", n_annotations=2,
                     agreement=0.8),
             OOSItem(item_id="b", question="q", label="in_scope", n_annotations=1),
             OOSItem(item_id="c", question="q", label="out_of_scope",
                     n_annotations=1)]
    write_gold_set("oos", items, tmp_path)
    stats = describe_gold_set("oos", items, tmp_path)
    assert stats.n == 3
    assert stats.label_counts == {"in_scope": 2, "out_of_scope": 1}
    assert stats.n_with_duplicate_annotation == 1
    assert stats.duplication_rate_achieved == round(1 / 3, 4)
    assert stats.duplication_rate_target == 0.30
    assert stats.mean_agreement == 0.8
    assert stats.n_with_agreement == 1


def test_describe_summarises_graded_retrieval_pools_by_grade():
    stats = describe_gold_set("retrieval", load_gold_set("retrieval"))
    assert all(k.startswith("grade_") for k in stats.label_counts)


def test_power_calculation_matches_the_research_protocol():
    """RESEARCH.md §7.1.4: n~=300 -> ~+/-5%, n~=500 -> ~+/-4%."""
    assert 0.055 < proportion_ci_halfwidth(300) < 0.058
    assert 0.043 < proportion_ci_halfwidth(500) < 0.045
    # Diminishing returns: 200 more items past 300 buys ~1.3pp.
    gain = proportion_ci_halfwidth(300) - proportion_ci_halfwidth(500)
    assert 0.010 < gain < 0.015


def test_power_calculation_shrinks_with_n():
    widths = [proportion_ci_halfwidth(n) for n in (50, 100, 300, 1000)]
    assert widths == sorted(widths, reverse=True)


def test_power_calculation_of_nothing_is_none_not_zero():
    assert proportion_ci_halfwidth(0) is None
    assert proportion_ci_halfwidth(-5) is None


def test_sizing_note_states_the_shortfall_in_numbers():
    stats = describe_gold_set("oos", load_gold_set("oos"))
    note = sizing_note(stats)
    assert "BELOW target" in note
    assert str(config.TARGET_N["oos"]) in note
    assert "95% CI" in note


def test_results_manifest_can_key_on_the_checksum():
    """The retention story depends on results files naming the exact gold data."""
    stats = describe_gold_set("oos", load_gold_set("oos"))
    raw = gold_path("oos").read_bytes()
    import hashlib
    assert stats.sha256 == hashlib.sha256(raw).hexdigest()


def test_shipped_files_are_valid_jsonl_one_object_per_line():
    for name in GOLD_SET_NAMES:
        for line in gold_path(name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                assert isinstance(json.loads(line), dict)
