# M6 persistent storage — validation evidence

What was actually run, and what it actually proves. Companion to
`docs/M6_PERSISTENT_STORAGE.md` (architecture) and
`docs/M6_STREAMLIT_CLOUD_DEPLOYMENT.md` (base deployment).

**Honest scope statement up front**: this session has no real GCS bucket or
credentials. Everything below that touches "GCS behavior" is verified
against an in-memory fake bucket (`FakeBucket`/`FakeBlob` in
`tests/test_cloud_storage.py`) that implements the exact subset of the real
`google.cloud.storage` API `GCSAnnotationBackend` calls
(`bucket.blob()`, `blob.upload_from_string()`, `blob.download_as_text()`,
`bucket.list_blobs(prefix=)`), plus code review of that call surface against
the already-working usage in `src/smartlawai/adapters/gcloud.py`. This proves
the logic (object-per-annotation, sync, backend selection) is correct; it
does **not** prove a real bucket/IAM/credentials setup works end-to-end —
that step still needs a human with real GCP access, per `CLAUDE.md`/`OPS.md`.

---

## 1. New + existing annotation-store test suites

```
$ .venv/Scripts/python -m pytest tests/test_annotation_store.py tests/test_cloud_storage.py -v
```

```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1, pluggy-1.6.0
collected 61 items

tests/test_annotation_store.py::test_blinding_strips_every_system_identifying_field PASSED [  1%]
tests/test_annotation_store.py::test_blinding_recurses_into_nested_dicts PASSED [  3%]
tests/test_annotation_store.py::test_blinding_keeps_the_content_the_rater_needs PASSED [  4%]
tests/test_annotation_store.py::test_blinded_field_list_covers_the_obvious_leaks PASSED [  6%]
tests/test_annotation_store.py::test_order_is_randomised_away_from_corpus_order PASSED [  8%]
tests/test_annotation_store.py::test_order_is_a_permutation_losing_nothing PASSED [  9%]
tests/test_annotation_store.py::test_order_is_stable_for_one_rater_so_a_resumed_session_matches PASSED [ 11%]
tests/test_annotation_store.py::test_order_differs_between_raters PASSED [ 13%]
tests/test_annotation_store.py::test_order_seed_changes_the_permutation PASSED [ 14%]
tests/test_annotation_store.py::test_mode_assignment_is_stable_per_item PASSED [ 16%]
tests/test_annotation_store.py::test_cold_subset_is_roughly_the_configured_fraction PASSED [ 18%]
tests/test_annotation_store.py::test_cold_assignment_is_item_keyed_so_raters_see_the_same_subset PASSED [ 19%]
tests/test_annotation_store.py::test_mode_fraction_extremes PASSED       [ 21%]
tests/test_annotation_store.py::test_invalid_cold_fraction_is_rejected PASSED [ 22%]
tests/test_annotation_store.py::test_append_and_reload_round_trip PASSED [ 24%]
tests/test_annotation_store.py::test_empty_store_loads_as_empty_list PASSED [ 26%]
tests/test_annotation_store.py::test_time_per_item_is_recorded_for_every_submission PASSED [ 27%]
tests/test_annotation_store.py::test_throughput_is_reported_per_mode_so_the_2x_claim_is_checkable PASSED [ 29%]
tests/test_annotation_store.py::test_throughput_without_timings_is_none_not_zero PASSED [ 31%]
tests/test_annotation_store.py::test_correction_mode_records_whether_the_proposal_was_accepted PASSED [ 32%]
tests/test_annotation_store.py::test_cold_mode_records_no_acceptance_because_there_was_nothing_to_accept PASSED [ 34%]
tests/test_annotation_store.py::test_store_is_append_only_and_the_later_submission_wins PASSED [ 36%]
tests/test_annotation_store.py::test_rated_item_ids_drives_the_apps_resume_queue PASSED [ 37%]
tests/test_annotation_store.py::test_store_rejects_an_unknown_task PASSED [ 39%]
tests/test_annotation_store.py::test_corrupt_log_line_names_its_line_number PASSED [ 40%]
tests/test_annotation_store.py::test_no_cloud_backend_by_default PASSED  [ 42%]
tests/test_annotation_store.py::test_append_uploads_to_a_configured_cloud_backend PASSED [ 44%]
tests/test_annotation_store.py::test_load_syncs_from_the_cloud_backend_before_reading PASSED [ 45%]
tests/test_annotation_store.py::test_explicit_none_cloud_disables_syncing_even_with_env_set PASSED [ 47%]
tests/test_annotation_store.py::test_live_agreement_from_two_raters PASSED [ 49%]
tests/test_annotation_store.py::test_live_agreement_unavailable_with_a_single_rater PASSED [ 50%]
tests/test_annotation_store.py::test_anchoring_pairs_compare_cold_labels_against_the_model_proposal PASSED [ 52%]
tests/test_annotation_store.py::test_items_with_no_proposal_anywhere_are_excluded_from_anchoring PASSED [ 54%]
tests/test_annotation_store.py::test_export_produces_a_schema_valid_gold_file PASSED [ 55%]
tests/test_annotation_store.py::test_export_aggregates_duplicated_items_into_one_row_with_agreement PASSED [ 57%]
tests/test_annotation_store.py::test_export_records_a_tie_rather_than_hiding_it PASSED [ 59%]
tests/test_annotation_store.py::test_export_majority_vote_over_three_raters PASSED [ 60%]
tests/test_annotation_store.py::test_export_honours_a_minimum_annotation_count PASSED [ 62%]
tests/test_annotation_store.py::test_export_skips_annotations_with_no_matching_item PASSED [ 63%]
tests/test_annotation_store.py::test_every_gold_set_has_a_rubric PASSED  [ 65%]
tests/test_annotation_store.py::test_rubric_options_match_the_schema_label_domain PASSED [ 67%]
tests/test_annotation_store.py::test_every_rubric_option_has_a_hotkey_and_a_description PASSED [ 68%]
tests/test_annotation_store.py::test_every_rubric_names_the_fields_the_rater_sees PASSED [ 70%]
tests/test_annotation_store.py::test_rubric_version_travels_with_every_rubric PASSED [ 72%]
tests/test_annotation_store.py::test_rubric_lookup_rejects_an_unknown_task PASSED [ 73%]
tests/test_cloud_storage.py::test_upload_annotation_writes_one_object_at_the_expected_path PASSED [ 75%]
tests/test_cloud_storage.py::test_upload_annotation_honours_a_custom_prefix PASSED [ 77%]
tests/test_cloud_storage.py::test_two_annotations_never_share_an_object PASSED [ 78%]
tests/test_cloud_storage.py::test_sync_to_local_pulls_new_objects_into_an_empty_file PASSED [ 80%]
tests/test_cloud_storage.py::test_sync_to_local_is_idempotent PASSED     [ 81%]
tests/test_cloud_storage.py::test_sync_to_local_only_appends_never_touches_existing_lines PASSED [ 83%]
tests/test_cloud_storage.py::test_sync_to_local_does_not_redownload_rows_already_present_locally PASSED [ 85%]
tests/test_cloud_storage.py::test_sync_to_local_on_a_missing_file_creates_it PASSED [ 86%]
tests/test_cloud_storage.py::test_sync_scopes_by_task_prefix PASSED      [ 88%]
tests/test_cloud_storage.py::test_concurrent_writers_to_the_same_item_both_survive PASSED [ 90%]
tests/test_cloud_storage.py::test_resolve_backend_defaults_to_local_when_unset PASSED [ 91%]
tests/test_cloud_storage.py::test_resolve_backend_local_is_explicit_no_op PASSED [ 93%]
tests/test_cloud_storage.py::test_resolve_backend_gcs_without_bucket_raises PASSED [ 95%]
tests/test_cloud_storage.py::test_resolve_backend_rejects_an_unknown_value PASSED [ 96%]
tests/test_cloud_storage.py::test_resolve_backend_gcs_uses_the_injected_bucket_factory_not_real_gcs PASSED [ 98%]
tests/test_cloud_storage.py::test_resolve_backend_honours_a_custom_gcs_prefix PASSED [100%]

============================= 61 passed in 0.28s ==============================
```

`tests/test_annotation_store.py` is the **pre-existing** M6 test file, run
here unmodified except for four new wiring tests appended at the bottom
(`test_no_cloud_backend_by_default`, `test_append_uploads_to_a_configured_cloud_backend`,
`test_load_syncs_from_the_cloud_backend_before_reading`,
`test_explicit_none_cloud_disables_syncing_even_with_env_set`) — every
original test in that file still passes as originally written, which is the
direct check for requirement 7 (backward compatibility / no regression in
local behavior).

`tests/test_cloud_storage.py::test_concurrent_writers_to_the_same_item_both_survive`
is the concurrency proof (requirement 6): two `GCSAnnotationBackend`
instances sharing one fake bucket, each writing a different annotation for
the same item, then a third `sync_to_local()` — asserts both submissions are
present. This demonstrates the "no silent overwrite" property structurally
(no shared mutable key exists to race over), not as a timing-dependent
probability that could still fail under real concurrent load.

## 2. Full project test suite

```
$ .venv/Scripts/python -m pytest -q
418 passed, 2 skipped in 18.20s
```

The 2 skips predate this change (an unrelated `bertscore_f1` test, skipped
because `bert-score`/`torch` are not installed in this environment — stated
as such in `tests/test_metrics.py` per `M6_ONBOARDING.md` §18). No new
skips or failures introduced anywhere in the project by this change.

## 3. Ruff

```
$ .venv/Scripts/python -m ruff check eval/annotate/cloud_storage.py eval/annotate/app.py \
    eval/annotate/store.py tests/test_cloud_storage.py tests/test_annotation_store.py
```

New code (`cloud_storage.py`, the `app.py` credentials-bridge addition,
`test_cloud_storage.py`) is clean. The 5 remaining findings are all
`RUF100 Unused noqa directive (non-enabled: E402)` on **pre-existing**
`# noqa: E402` import-order comments in `app.py` that predate this change
(this repo's `pyproject.toml` `[tool.ruff]` section does not select `E402`,
so those old comments are now flagged as unused by a newer ruff — unrelated
to this feature, left as-is per "do not modify unrelated functionality"). A
full-repo `ruff check .` shows 95 pre-existing findings across the project
for the same reason — this change did not add to that count.

## 4. Local app smoke test (manual, via browser)

Ran the actual Streamlit app locally (`streamlit run eval/annotate/app.py`,
no `SMARTLAW_ANNOTATIONS_BACKEND` set — the default path):

1. Loaded the app, entered rater ID `restart-check`, opened the `retrieval`
   gold set, submitted one annotation. Sidebar showed "1 annotations
   submitted."
2. **Killed the Streamlit process** (simulating a Cloud restart/redeploy —
   the actual failure mode this feature exists to survive).
3. Read `eval/annotations/retrieval.jsonl` directly off disk: the submitted
   row was there, byte-identical to what `append()` would have written
   before this change (confirms the local write path is unmodified).
4. **Restarted** the Streamlit process fresh and reopened the app with the
   same rater ID. The app correctly showed "1 / 10 items done by this
   rater" and "1 annotations submitted," and resumed the queue at the next
   unrated item — proving the local (non-cloud) durability path that
   already existed continues to work exactly as before, and that nothing in
   this change broke the resume-on-reload flow `app.py` depends on.
5. Also confirmed the new `_bridge_gcp_credentials_from_secrets()` function
   in `app.py` is inert with no `.streamlit/secrets.toml` present — no
   exception, no behavior change, app loads identically to before this
   change (this is what requirement 1, "must work locally without cloud
   credentials," actually depends on).
6. Test annotation rows were deleted from `eval/annotations/` afterward
   (that directory is gitignored, so this only affected local disk, never
   git).

## 5. What was NOT verified (explicitly)

- **No live GCS bucket exists yet** — bucket creation is a human/`gcloud`
  step per `CLAUDE.md`/`OPS.md` (drafted commands are in
  `docs/M6_PERSISTENT_STORAGE.md`, not run by this agent). So
  `resolve_backend()`'s real path (`_real_gcs_bucket()` →
  `google.cloud.storage.Client()` → a real network call) has not been
  exercised end-to-end — only its wiring (env-var parsing, error cases, and
  the call into an injected `bucket_factory`) is tested, per
  `tests/test_cloud_storage.py::test_resolve_backend_gcs_uses_the_injected_bucket_factory_not_real_gcs`.
- **No Streamlit Cloud deployment** — `_bridge_gcp_credentials_from_secrets()`
  has not been exercised against Cloud's actual Secrets manager, only
  reasoned through against Streamlit's documented secrets API and confirmed
  inert (does nothing, raises nothing) when secrets aren't present, which is
  what running it locally proves.
- Recommendation before trusting real annotators to it: after the bucket and
  service account exist, do one manual end-to-end check (submit an
  annotation on the deployed Cloud app, confirm the object appears in the
  bucket via `gcloud storage ls`, redeploy, confirm the app still reports
  the annotation as submitted) before wider annotator use. This is a fast
  check but does require the real GCP resources this session doesn't have.
