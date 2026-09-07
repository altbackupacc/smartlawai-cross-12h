# M6 Implementation Plan

Companion to `docs/M6_IMPLEMENTATION_ANALYSIS.md`. Fourteen phases, each independently
verifiable. The ordering follows `M6_ONBOARDING.md` §17 and reference image 2's phase
workflow, with one deliberate deviation noted in Phase 1.

Standing rules for every phase:

- `pytest -q` stays green. A phase is not done if it broke an existing test.
- No magic numbers outside `eval/config.py` (CLAUDE.md §4 applied to the eval package;
  `src/smartlawai/config.py` stays untouched because eval thresholds are not pipeline
  thresholds and mixing them would put eval concerns in serving code).
- I2 everywhere: a metric that cannot be computed returns `None` with a stated reason.
  Never a substituted number.
- Nothing in `eval/` except `run_eval.py` imports `smartlawai`.

---

## Phase 1 — Architecture, package skeleton, configuration, and the §4 move

**Purpose.** Create the top-level `eval/` package `PLAN.md` specifies, resolve the
pre-existing `src/smartlawai/eval/metrics.py` misplacement, and centralise every eval
threshold.

**Deviation from the onboarding's ordering, stated explicitly:** §17 puts the annotation
app at step 4 and the move at step 1. The app cannot be built before the schemas and
storage it writes, so Phases 2–4 (schemas, gold-set IO, guidelines/storage) precede the
app in Phase 10. The *substance* of "build the app first" — that gold data collection is
not an afterthought bolted on at the end — is preserved: every phase before 10 exists to
serve the app, and no gold set is populated before the app that produces it exists.

**Files to create:** `eval/__init__.py`, `eval/config.py`, `eval/README.md`,
`eval/results/.gitkeep`.
**Files to modify:** `src/smartlawai/eval/metrics.py` -> `eval/metrics.py` (git mv
semantics); delete `src/smartlawai/eval/`; `tests/test_metrics.py` and
`tests/test_local_backend.py` import lines; `conftest.py` (repo root on `sys.path`);
`pyproject.toml` (`[eval]` extras); `.gitignore`.
**Dependencies:** none.
**Inputs:** existing repo.
**Outputs:** importable `eval` package; `import eval.metrics` works from tests.
**Tests required:** the existing `tests/test_metrics.py` passes from the new path.
**Definition of done:** `pytest -q` green at the pre-existing count; `src/smartlawai/eval/`
gone; no file anywhere still imports `smartlawai.eval`.

## Phase 2 — Evaluation data schemas

**Purpose.** One typed, versioned definition per gold set, plus the annotation record,
so the app, the loader, the runner and the tests all agree without duplicated field lists.

**Files to create:** `eval/schemas.py`.
**Files to modify:** none.
**Dependencies:** Phase 1.
**Inputs:** `RESEARCH.md` §7.1's set table; `protocols.py`'s `Claim` field names.
**Outputs:** dataclasses `RetrievalItem`, `OOSItem`, `GroundednessItem`, `RepealedItem`,
`EntailmentItem`, `Annotation`, plus `GOLD_SET_SPECS` (name, schema, IAA duplication rate,
target N, label set) and `validate_row()` / `to_row()` / `from_row()`.
**Tests required:** `tests/test_eval_schemas.py` — round-trip, required-field rejection,
label-domain rejection, IAA rate table matches `RESEARCH.md` §7.1.3.
**Definition of done:** every one of the five sets has a schema whose label domain is
enumerated in code, not in a docstring.

## Phase 3 — Gold test set infrastructure

**Purpose.** Load, validate, freeze and describe gold sets. Frozen means a checksum is
recorded, so silent edits are detectable (I5-adjacent discipline).

**Files to create:** `eval/gold_sets.py`, `eval/gold/PROVENANCE.md`,
`eval/gold/external/PROVENANCE.md`, the five (initially seed-populated) `.jsonl` files.
**Files to modify:** none.
**Dependencies:** Phase 2.
**Inputs:** `eval/gold/*.jsonl`.
**Outputs:** `load_gold_set(name)` -> validated items; `GoldSetStats` (n, label
distribution, duplicate-annotation coverage, sha256); `power_ci_halfwidth(n, p)` used to
state the sizing calculation per set; graceful, explicit behaviour when a file is absent.
**Tests required:** `tests/test_gold_sets.py` — loads each shipped set, rejects a
malformed row with the offending line number, missing file raises a stated error rather
than returning an empty list that reads as "zero items measured".
**Definition of done:** all five files load and validate; each has a provenance entry
stating what it is, how it was produced, and its power calculation.

## Phase 4 — Annotation guidelines and annotation storage

**Purpose.** Rubrics as data (image 2's "guidelines & rating scales" panel) and an
append-only annotation log that records rater id, timestamp and time-per-item.

**Files to create:** `eval/annotate/__init__.py`, `eval/annotate/guidelines.py`,
`eval/annotate/store.py`.
**Files to modify:** none.
**Dependencies:** Phase 2.
**Inputs:** `RESEARCH.md` §7; `PLAN.md` M6's app requirements.
**Outputs:** `RUBRICS` (per-task label set, scale, per-label description, worked
examples, version string); `AnnotationStore` with `append()`, `load()`, `by_item()`,
`export_gold_set()`, `items_per_hour()`; JSONL on disk, no database.
**Tests required:** `tests/test_annotation_store.py` — append/reload round-trip,
time-per-item recorded, export produces schema-valid gold rows, export of a duplicated
item aggregates rather than emitting two rows, rubric version travels into the export.
**Definition of done:** an annotation written by the app can be exported to a gold file
that Phase 3's loader validates, with no manual step.

## Phase 5 — Evaluation runner

**Purpose.** The orchestrator: gold set in, `Pipeline` called, per-item records out.

**Files to create:** `eval/run_eval.py` (runner core; metric wiring lands in Phases 6–9).
**Files to modify:** none.
**Dependencies:** Phases 2–3.
**Inputs:** `--split`, `--gold-dir`, `--out-dir`, `--seed`, `--corpus`.
**Outputs:** per-item `ItemResult` (question, decision, answer, retrieved ids,
verification statuses/scores, stage timings, cost) and a `RunManifest`
(timestamp, seed, git commit, system config, gold-set checksums, package versions).
**Tests required:** `tests/test_run_eval.py` — runs against a temp backend and the
sample judgment; asserts `Pipeline` is called with an explicit `Scope` (I1) and that
eval never writes into a real local store.
**Definition of done:** the runner executes end to end on stub models and emits per-item
records; no metric is hand-typed.

## Phase 6 — Retrieval evaluation metrics

**Purpose.** Close the `RESEARCH.md` §5 retrieval row: Recall@{5,10,20}, MRR, nDCG@10,
Precision@K.

**Files to create:** none.
**Files to modify:** `eval/metrics.py` (add `ndcg_at_k`, `retrieval_metrics` aggregator).
**Dependencies:** Phase 1.
**Inputs:** ranked chunk ids from the trace; gold relevant ids (graded where present).
**Outputs:** metric dict; `None` with a reason when a query has no judged relevant chunk.
**Tests required:** extended `tests/test_metrics.py` — nDCG against a hand-computed
worked example, perfect/reversed/empty rankings, graded vs binary relevance.
**Definition of done:** nDCG@10 matches an independently computed value on a fixture.

## Phase 7 — Answer quality and groundedness evaluation

**Purpose.** ROUGE via the real library, BERTScore kept, extractiveness added,
claim-level groundedness (the project's single hallucination definition, CLAUDE.md §5).

**Files to create:** `eval/extractiveness.py`.
**Files to modify:** `eval/metrics.py` (delete `rouge_l`, add `rouge_scores`,
`groundedness_rate`, `hallucination_rate`), `tests/test_metrics.py`.
**Dependencies:** Phases 1–2.
**Inputs:** prediction/reference strings; `(claim, passage, supported)` triples.
**Outputs:** ROUGE-1/2/L F1; extractiveness in [0,1]; groundedness and hallucination
rates that are exact complements of one another by construction.
**Tests required:** `tests/test_extractiveness.py` (verbatim copy -> 1.0, disjoint -> 0.0,
short-text edge cases, n-gram window shorter than the text); ROUGE exact/partial/empty
cases retargeted from `rouge_l`; `bertscore_f1` under `importorskip`.
**Definition of done:** no hand-rolled ROUGE remains anywhere in the repo;
`hallucination_rate == 1 - groundedness_rate` is asserted, enforcing CLAUDE.md §5's
"one definition only".

## Phase 8 — Citation / authority evaluation

**Purpose.** Citation coverage (exists), plus `RESEARCH.md` §5's citation-validity and
repealed-citation rates, without creating a dependency on M4.

**Files to create:** none.
**Files to modify:** `eval/metrics.py` (`citation_validity_rate`,
`repealed_citation_rate`, `citation_support_rate`).
**Dependencies:** Phase 1.
**Inputs:** lists of already-resolved registry-result dicts
`{citation, exists, in_force, superseded_by}`.
**Outputs:** rates, or `None` + `"registry_unavailable"` when resolution is absent — the
I2 behaviour, never a 0.0 that reads as "no repealed citations found".
**Tests required:** extended `tests/test_metrics.py` — mixed in-force/repealed input,
all-unresolved input returns `None`, empty input returns `None` not `0.0`.
**Definition of done:** `grep -r "registry\|verify" eval/` shows no import of M4/M5 code.

## Phase 9 — Out-of-scope, robustness, and perturbation evaluation

**Purpose.** Refusal precision/recall on the labelled OOS set; per-perturbation-type
detection rate; the perturbation generator itself.

**Files to create:** `data/perturbations/generate_perturbations.py`,
`data/perturbations/README.md`.
**Files to modify:** `eval/metrics.py` (`refusal_metrics`,
`per_perturbation_detection_rate`), `.gitignore`.
**Dependencies:** Phases 2–3.
**Inputs:** IN-Abs judgement/summary text pairs; gold OOS labels; system decisions.
**Outputs:** `perturbations.jsonl` rows `{pair_id, source_doc_id, source_split,
perturbation_type, passage, claim, label, seed}` across all seven `RESEARCH.md` §2.5
types; refusal P/R/F1; per-type detection rates.
**Tests required:** `tests/test_perturbations.py` — each of the seven types actually
changes the claim; determinism under a fixed seed; every row carries `source_doc_id`;
disjointness from `eval/gold/*.jsonl`; a `pytest.skip`-guarded train-split-only check
that becomes a real assertion the moment `data/splits/*.json` exists.
**Definition of done:** the generator runs standalone (`python
data/perturbations/generate_perturbations.py --n 50`), is seeded and reproducible, and
the disjointness test is enforced (not skipped) against the shipped gold sets.

## Phase 10 — Human annotation interface

**Purpose.** The Streamlit app `PLAN.md` calls the prerequisite for the gold sets.

**Files to create:** `eval/annotate/app.py`, `eval/annotate/pooling.py`.
**Files to modify:** none.
**Dependencies:** Phases 2–4.
**Inputs:** a task name, a rater id, a queue file, a mode (`correction` / `cold`).
**Outputs:** annotations appended to the store; live agreement in the sidebar; one-click
export to `eval/gold/*.jsonl`.
**Requirements enforced in code, not by convention:** randomised item order seeded per
rater; system identifiers stripped before display (blinding); correction vs cold mode
with a 10% cold subset selected deterministically; rater id, timestamp and time-per-item
logged on every submission; live Cohen's kappa / Krippendorff's alpha; rubric and scale
rendered on screen; nothing in the schema names the annotator pool.
**Tests required:** `tests/test_pooling.py` (TREC pooling: dedup across methods, top-10
per method, ~15 unique candidates, deterministic order); the app's non-UI helpers
(`blind_item`, `randomised_order`, `assign_mode`) are pure functions tested in
`tests/test_annotation_store.py`. Streamlit rendering itself is not unit-tested; the app
is import-guarded so a missing Streamlit is a clear message, not a stack trace.
**Definition of done:** `streamlit run eval/annotate/app.py` opens, labels an item, and
the item appears in the store with a nonzero `seconds_on_item`.

## Phase 11 — Statistical analysis

**Purpose.** `RESEARCH.md` §6's protocol as code.

**Files to create:** `eval/stats.py`.
**Files to modify:** none.
**Dependencies:** Phase 1.
**Inputs:** paired per-item scores; confidence/correctness pairs; p-values.
**Outputs:** `paired_bootstrap` (10k resamples, seed 42, mean difference + 95% CI +
`significant` flag), `expected_calibration_error`, `reliability_diagram_data`,
`bonferroni_correct`, `mean_std` (the "mean +/- std always" reporting helper),
`proportion_ci_halfwidth` (the power calculation §7.1.4 demands).
**Tests required:** `tests/test_stats.py` — bootstrap CI covers a known difference and
excludes zero when it should; identical inputs give a CI containing zero; ECE is 0 for
perfectly calibrated input and 1 for maximally miscalibrated; reliability bins sum to n;
Bonferroni rejects fewer hypotheses than uncorrected; determinism under a fixed seed.
**Definition of done:** every number `RESEARCH.md` §6 asks the paper to report has a
function that produces it.

## Phase 12 — Evaluation reporting

**Purpose.** Image 2's "evaluation report (tables + results)": machine-readable JSON plus
a human-readable Markdown table.

**Files to create:** `eval/report.py`.
**Files to modify:** `eval/run_eval.py` (call it).
**Dependencies:** Phases 5–11.
**Inputs:** the runner's results dict.
**Outputs:** `eval/results/{split}_{timestamp}.json` and `.md` containing dataset info,
n, system configuration, retrieval/answer/citation/robustness metric tables, human-eval
summary when annotations exist, failure list, error analysis by failure reason, and an
explicit `unavailable` section naming every metric that could not be computed and why.
**Tests required:** `tests/test_report.py` — report renders from a fixture; unavailable
metrics appear as `unavailable` with a reason, never as `0.0`; JSON and Markdown agree.
**Definition of done:** a reader of the Markdown can tell which numbers are measured and
which are missing, without opening the JSON.

## Phase 13 — End-to-end integration with SmartLawAI

**Purpose.** The done-when criterion: the full table computes for the untrained system.

**Files to create:** `.github/workflows/eval-regression.yml`.
**Files to modify:** `eval/run_eval.py` (`--baseline` comparison + nonzero exit on
regression).
**Dependencies:** Phases 5–12.
**Inputs:** `eval/gold/*.jsonl`, the stub pipeline, the last committed
`eval/results/dev_*.json`.
**Outputs:** a committed dev results file; a CI job that fails on regression past a
generous threshold.
**Tests required:** `tests/test_eval_integration.py` — full run over all five sets on the
stub pipeline; asserts the report is written, every gold set is represented, I1 holds
(every ask carries a Scope), and I2 holds (no unavailable component became a number).
**Definition of done:** `python -m eval.run_eval --split dev` writes a complete results
table with a zero exit code.

## Phase 14 — Testing, validation, and documentation

**Purpose.** Prove nothing else broke, and leave the milestone legible.

**Files to create:** `docs/M6_COMPLETION_REPORT.md`.
**Files to modify:** `eval/README.md`, `PLAN.md` progress line if warranted.
**Dependencies:** all.
**Outputs:** full `pytest -q` run, `ruff check` clean at the configured line length, a
dead-code and duplicate-functionality review, and the completion report.
**Definition of done:** every pre-existing test still passes, all new tests pass, ruff is
clean on every file M6 touched, and the report states what was built, what was seeded
rather than annotated, and what a real annotation study still has to do.
