"""Annotation storage and the app's methodological guarantees (M6 Phases 4/10).

The pure helpers tested here -- blinding, randomisation, mode assignment --
carry the properties PLAN.md M6 calls non-negotiable. They live outside app.py
precisely so they can be asserted rather than eyeballed in a browser.
"""
from __future__ import annotations

import pytest

from eval import config
from eval.annotate.guidelines import RUBRIC_VERSION, RUBRICS, rubric_for
from eval.annotate.store import (
    BLINDED_FIELDS,
    AnnotationStore,
    assign_mode,
    blind_item,
    randomised_order,
)
from eval.gold_sets import load_gold_set
from eval.schemas import GOLD_SET_NAMES, OOSItem, SchemaError


@pytest.fixture()
def store(tmp_path):
    return AnnotationStore("oos", root=tmp_path)


# --------------------------------------------------------------------------- #
# Blinding
# --------------------------------------------------------------------------- #
def test_blinding_strips_every_system_identifying_field():
    item = {"item_id": "x", "question": "q", "system": "ours", "model_id": "m-1",
            "is_ours": True, "arm": "L"}
    blinded = blind_item(item)
    assert blinded == {"item_id": "x", "question": "q"}


def test_blinding_recurses_into_nested_dicts():
    """A model id hides inside a detail or trace blob, which is exactly where a
    rater would spot it."""
    item = {"item_id": "x", "detail": {"model": "mistral-7b", "score": 0.9}}
    assert blind_item(item)["detail"] == {"score": 0.9}


def test_blinding_keeps_the_content_the_rater_needs():
    item = {"claim_text": "c", "passage": "p", "system_name": "baseline"}
    blinded = blind_item(item)
    assert blinded["claim_text"] == "c" and blinded["passage"] == "p"
    assert "system_name" not in blinded


def test_blinded_field_list_covers_the_obvious_leaks():
    assert {"system", "model", "model_id", "is_ours", "arm"} <= BLINDED_FIELDS


# --------------------------------------------------------------------------- #
# Randomisation
# --------------------------------------------------------------------------- #
def test_order_is_randomised_away_from_corpus_order():
    ids = [f"i{i:03d}" for i in range(60)]
    assert randomised_order(ids, "rater-1") != ids


def test_order_is_a_permutation_losing_nothing():
    ids = [f"i{i}" for i in range(30)]
    assert sorted(randomised_order(ids, "r1")) == sorted(ids)


def test_order_is_stable_for_one_rater_so_a_resumed_session_matches():
    ids = [f"i{i}" for i in range(30)]
    assert randomised_order(ids, "r1") == randomised_order(ids, "r1")


def test_order_differs_between_raters():
    ids = [f"i{i}" for i in range(30)]
    assert randomised_order(ids, "r1") != randomised_order(ids, "r2")


def test_order_seed_changes_the_permutation():
    ids = [f"i{i}" for i in range(30)]
    assert randomised_order(ids, "r1", seed=1) != randomised_order(ids, "r1", seed=2)


# --------------------------------------------------------------------------- #
# Cold / correction mode (anchoring control)
# --------------------------------------------------------------------------- #
def test_mode_assignment_is_stable_per_item():
    assert assign_mode("item-42") == assign_mode("item-42")


def test_cold_subset_is_roughly_the_configured_fraction():
    """RESEARCH.md §7.1.1 wants ~10% labelled cold."""
    modes = [assign_mode(f"item-{i}") for i in range(2000)]
    cold_fraction = modes.count("cold") / len(modes)
    assert 0.07 < cold_fraction < 0.13
    assert config.COLD_SUBSET_FRACTION == 0.10


def test_cold_assignment_is_item_keyed_so_raters_see_the_same_subset():
    """Comparability across raters is the whole point of the control."""
    ids = [f"item-{i}" for i in range(200)]
    assert [assign_mode(i) for i in ids] == [assign_mode(i) for i in ids]


def test_mode_fraction_extremes():
    assert assign_mode("x", cold_fraction=0.0) == "correction"
    assert assign_mode("x", cold_fraction=1.0) == "cold"


def test_invalid_cold_fraction_is_rejected():
    with pytest.raises(ValueError):
        assign_mode("x", cold_fraction=1.5)


# --------------------------------------------------------------------------- #
# The log
# --------------------------------------------------------------------------- #
def test_append_and_reload_round_trip(store):
    store.record("i1", "r1", "in_scope", "correction", 5.0,
                 proposed_label="in_scope", rubric_version=RUBRIC_VERSION)
    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].rater_id == "r1"
    assert loaded[0].seconds_on_item == 5.0
    assert loaded[0].rubric_version == RUBRIC_VERSION


def test_empty_store_loads_as_empty_list(store):
    assert store.load() == []


def test_time_per_item_is_recorded_for_every_submission(store):
    """RESEARCH.md §7.3: this number sizes every remaining gold set."""
    store.record("i1", "r1", "in_scope", "cold", 12.0)
    store.record("i2", "r1", "out_of_scope", "cold", 24.0)
    t = store.throughput()
    assert t.n_annotations == 2
    assert t.mean_seconds == 18.0
    assert t.median_seconds == 18.0
    assert t.items_per_hour == pytest.approx(200.0, abs=0.1)


def test_throughput_is_reported_per_mode_so_the_2x_claim_is_checkable(store):
    """RESEARCH.md §7.1.1 claims 2-3x throughput from model-assisted labelling."""
    for i in range(3):
        store.record(f"c{i}", "r1", "in_scope", "correction", 10.0,
                     proposed_label="in_scope")
    for i in range(3):
        store.record(f"k{i}", "r1", "in_scope", "cold", 30.0)
    by_mode = store.throughput().by_mode
    assert by_mode["correction"] > by_mode["cold"]


def test_throughput_without_timings_is_none_not_zero(store):
    store.record("i1", "r1", "in_scope", "cold", 0.0)
    assert store.throughput().items_per_hour is None


def test_correction_mode_records_whether_the_proposal_was_accepted(store):
    store.record("i1", "r1", "in_scope", "correction", 3.0,
                 proposed_label="in_scope")
    store.record("i2", "r1", "out_of_scope", "correction", 3.0,
                 proposed_label="in_scope")
    accepted = [a.accepted_proposal for a in store.load()]
    assert accepted == [True, False]


def test_cold_mode_records_no_acceptance_because_there_was_nothing_to_accept(store):
    store.record("i1", "r1", "in_scope", "cold", 3.0)
    assert store.load()[0].accepted_proposal is None


def test_store_is_append_only_and_the_later_submission_wins(store):
    store.record("i1", "r1", "in_scope", "cold", 3.0)
    store.record("i1", "r1", "out_of_scope", "cold", 4.0)
    assert len(store.load()) == 2                       # nothing was overwritten
    latest = store.latest_by_rater()["i1"]["r1"]
    assert latest.label == "out_of_scope"


def test_rated_item_ids_drives_the_apps_resume_queue(store):
    store.record("i1", "r1", "in_scope", "cold", 3.0)
    assert store.rated_item_ids("r1") == {"i1"}
    assert store.rated_item_ids("r2") == set()


def test_store_rejects_an_unknown_task(tmp_path):
    with pytest.raises(SchemaError, match="unknown task"):
        AnnotationStore("not-a-set", root=tmp_path)


def test_corrupt_log_line_names_its_line_number(store):
    store.record("i1", "r1", "in_scope", "cold", 3.0)
    with store.path.open("a", encoding="utf-8") as fh:
        fh.write("{broken\n")
    with pytest.raises(SchemaError, match=r":2:"):
        store.load()


# --------------------------------------------------------------------------- #
# Cloud backend wiring (docs/M6_PERSISTENT_STORAGE.md). The GCS semantics
# themselves -- object-per-annotation, sync-to-local, backend selection --
# are covered in tests/test_cloud_storage.py against a fake bucket; these
# only prove AnnotationStore actually calls into whatever backend it's given,
# and that the default (no backend configured) is unchanged from before.
# --------------------------------------------------------------------------- #
class _RecordingCloud:
    """Minimal stand-in for GCSAnnotationBackend, just to observe calls."""

    def __init__(self) -> None:
        self.uploaded: list[dict] = []
        self.synced: list[str] = []

    def upload_annotation(self, task, row):
        self.uploaded.append(row)

    def sync_to_local(self, task, local_path):
        self.synced.append(task)


def test_no_cloud_backend_by_default(tmp_path):
    """SMARTLAW_ANNOTATIONS_BACKEND is unset in the test environment, so the
    default `cloud="auto"` resolves to None -- local-only, same as before
    persistent storage existed."""
    store = AnnotationStore("oos", root=tmp_path)
    assert store._cloud is None


def test_append_uploads_to_a_configured_cloud_backend(tmp_path):
    cloud = _RecordingCloud()
    store = AnnotationStore("oos", root=tmp_path, cloud=cloud)
    store.record("i1", "r1", "in_scope", "cold", 3.0)

    assert len(cloud.uploaded) == 1
    assert cloud.uploaded[0]["item_id"] == "i1"
    # the local write still happens too -- cloud is additive, not a replacement
    assert len(store.load()) == 1


def test_load_syncs_from_the_cloud_backend_before_reading(tmp_path):
    cloud = _RecordingCloud()
    store = AnnotationStore("oos", root=tmp_path, cloud=cloud)
    store.load()
    assert cloud.synced == ["oos"]


def test_explicit_none_cloud_disables_syncing_even_with_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTLAW_ANNOTATIONS_BACKEND", "gcs")
    monkeypatch.setenv("GCS_BUCKET", "smartlawai-annotations")
    # cloud=None wins over env -- proves "auto" is opt-in, not the only path.
    store = AnnotationStore("oos", root=tmp_path, cloud=None)
    store.record("i1", "r1", "in_scope", "cold", 3.0)
    assert store.load()[0].item_id == "i1"   # no attempt to reach real GCS


# --------------------------------------------------------------------------- #
# Live agreement and anchoring from the log
# --------------------------------------------------------------------------- #
def test_live_agreement_from_two_raters(store):
    for item, (a, b) in {"i1": ("in_scope", "in_scope"),
                         "i2": ("out_of_scope", "out_of_scope"),
                         "i3": ("in_scope", "advice_seeking")}.items():
        store.record(item, "r1", a, "cold", 3.0)
        store.record(item, "r2", b, "cold", 3.0)
    r = store.agreement()
    assert r.statistic == "cohen_kappa"
    assert r.value is not None


def test_live_agreement_unavailable_with_a_single_rater(store):
    store.record("i1", "r1", "in_scope", "cold", 3.0)
    assert store.agreement().value is None


def test_anchoring_pairs_compare_cold_labels_against_the_model_proposal(store):
    store.record("i1", "r1", "in_scope", "correction", 3.0,
                 proposed_label="in_scope")
    store.record("i1", "r2", "out_of_scope", "cold", 3.0)
    assisted, cold = store.anchoring_pairs()
    assert assisted == [("in_scope", "in_scope")]
    assert cold == [("out_of_scope", "in_scope")]


def test_items_with_no_proposal_anywhere_are_excluded_from_anchoring(store):
    store.record("i1", "r1", "in_scope", "cold", 3.0)
    assert store.anchoring_pairs() == ([], [])


# --------------------------------------------------------------------------- #
# Export -- PLAN.md M6: "straight to eval/gold/*.jsonl, no spreadsheet step"
# --------------------------------------------------------------------------- #
def test_export_produces_a_schema_valid_gold_file(tmp_path):
    store = AnnotationStore("oos", root=tmp_path)
    items = {"i1": OOSItem(item_id="i1", question="Should I sue?",
                           label="in_scope")}
    store.record("i1", "r1", "advice_seeking", "cold", 4.0,
                 rubric_version=RUBRIC_VERSION)
    exported = store.export_gold_set(items, gold_dir=tmp_path)

    assert len(exported) == 1
    assert exported[0].label == "advice_seeking"     # the rater's label, not the seed
    assert exported[0].rater_ids == ["r1"]
    assert exported[0].n_annotations == 1
    assert exported[0].rubric_version == RUBRIC_VERSION
    assert load_gold_set("oos", tmp_path)[0].label == "advice_seeking"


def test_export_aggregates_duplicated_items_into_one_row_with_agreement(tmp_path):
    store = AnnotationStore("oos", root=tmp_path)
    items = {"i1": OOSItem(item_id="i1", question="q", label="in_scope"),
             "i2": OOSItem(item_id="i2", question="q2", label="in_scope")}
    for item, (a, b) in {"i1": ("in_scope", "in_scope"),
                         "i2": ("out_of_scope", "in_scope")}.items():
        store.record(item, "r1", a, "cold", 3.0)
        store.record(item, "r2", b, "cold", 3.0)
    exported = store.export_gold_set(items, gold_dir=tmp_path)

    assert len(exported) == 2                       # two raters, still two rows
    assert all(i.n_annotations == 2 for i in exported)
    assert all(sorted(i.rater_ids) == ["r1", "r2"] for i in exported)


def test_export_records_a_tie_rather_than_hiding_it(tmp_path):
    """A tied item is the ambiguity RESEARCH.md §7.3's pilot exists to surface."""
    store = AnnotationStore("oos", root=tmp_path)
    items = {"i1": OOSItem(item_id="i1", question="q", label="in_scope")}
    store.record("i1", "r1", "in_scope", "cold", 3.0)
    store.record("i1", "r2", "out_of_scope", "cold", 3.0)
    exported = store.export_gold_set(items, gold_dir=tmp_path)
    assert "tie" in exported[0].notes


def test_export_majority_vote_over_three_raters(tmp_path):
    store = AnnotationStore("oos", root=tmp_path)
    items = {"i1": OOSItem(item_id="i1", question="q", label="in_scope")}
    for rater, label in (("r1", "advice_seeking"), ("r2", "advice_seeking"),
                         ("r3", "in_scope")):
        store.record("i1", rater, label, "cold", 3.0)
    exported = store.export_gold_set(items, gold_dir=tmp_path)
    assert exported[0].label == "advice_seeking"
    assert "tie" not in exported[0].notes


def test_export_honours_a_minimum_annotation_count(tmp_path):
    store = AnnotationStore("oos", root=tmp_path)
    items = {"i1": OOSItem(item_id="i1", question="q", label="in_scope")}
    store.record("i1", "r1", "in_scope", "cold", 3.0)
    assert store.export_gold_set(items, gold_dir=tmp_path, min_annotations=2) == []


def test_export_skips_annotations_with_no_matching_item(tmp_path):
    store = AnnotationStore("oos", root=tmp_path)
    store.record("ghost", "r1", "in_scope", "cold", 3.0)
    assert store.export_gold_set({}, gold_dir=tmp_path) == []


# --------------------------------------------------------------------------- #
# Rubrics as data
# --------------------------------------------------------------------------- #
def test_every_gold_set_has_a_rubric():
    assert set(RUBRICS) == set(GOLD_SET_NAMES)


def test_rubric_options_match_the_schema_label_domain():
    from eval.schemas import GOLD_SET_SPECS
    for name, rubric in RUBRICS.items():
        assert set(rubric.values()) == set(GOLD_SET_SPECS[name].labels), name


def test_every_rubric_option_has_a_hotkey_and_a_description():
    """One item per screen, keyboard-driven (PLAN.md M6)."""
    for rubric in RUBRICS.values():
        for option in rubric.options:
            assert option.hotkey and option.description


def test_every_rubric_names_the_fields_the_rater_sees():
    for rubric in RUBRICS.values():
        assert rubric.fields_shown


def test_rubric_version_travels_with_every_rubric():
    for rubric in RUBRICS.values():
        assert rubric.version == RUBRIC_VERSION


def test_rubric_lookup_rejects_an_unknown_task():
    with pytest.raises(KeyError):
        rubric_for("nope")
