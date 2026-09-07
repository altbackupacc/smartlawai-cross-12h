# M6 Gap Analysis

Companion to `docs/M6_REQUIREMENTS_TRACEABILITY.md`. Every gap below traces to a
specific traceability row. CRITICAL and HIGH gaps are fixed in this same audit
pass (Step 6); MEDIUM gaps are fixed where cheap and safe; LOW gaps are recorded
for awareness only.

---

## CRITICAL

### G1 — `score_groundedness` computed the gold set's own label distribution, not a system measurement

**Source requirement.** CLAUDE.md I2 ("never substitute a default score... that
bug caused fabricated metrics"), CLAUDE.md §5 (hallucination rate is
"judge-based"), RESEARCH.md §5 ("Faithfulness | HHEM-2.1, claim-level"; "QA |
Claim-level groundedness | InLegalNLI + HHEM-2.1, both reported").

**Current implementation (before fix).** `eval/run_eval.py::score_groundedness`
called `groundedness_rate([i.supported() for i in items])` — i.e. it read each
gold item's own human label and reported that distribution back as
`groundedness_rate`/`hallucination_rate`. For the shipped 40-item seed set (20
`entailed` / 20 `not_entailed` by construction), this produces exactly `0.5000`
every time, regardless of what any verifier does. Confirmed in the committed
`eval/results/dev_20260905T162402Z.md`.

**What's wrong.** This is not measuring the system. It is measuring the
dataset's own composition and presenting it as a computed metric with
`status: "ok"`. It is functionally the same category of error CLAUDE.md I2 names
as its motivating example (a verifier returning a default `0.5`) — a number that
looks measured but was never actually produced by evaluating anything. A reader
of the report cannot tell the difference between a real 50% groundedness rate
and this artifact.

**Corrective action.** Exercise the pipeline's real `Verifier` component (the
actual, existing, testable system artifact that stands in for the not-yet-built
HHEM/InLegalNLI) on each `(claim, passage)` pair, and compute
`groundedness_rate`/`hallucination_rate` from *its* predictions. Separately,
compute the legitimate "Judge | Agreement with human labels" metric (RESEARCH.md
§5) by comparing the verifier's prediction against the gold label via Cohen's
κ and raw accuracy — this is the number the gold set actually exists to produce.

**Files affected.** `eval/run_eval.py` (`score_groundedness` rewritten to accept
`corpus` and call `corpus.pipeline.verifier.verify(...)`), `tests/test_run_eval.py`
(tests rewritten to assert against verifier behaviour, not gold-label echo).

---

### G2 — `score_repealed` fabricated registry results from gold labels, and used the wrong `RegistryResult` shape entirely

**Source requirement.** M6_ONBOARDING §8 ("even though they can't be computed
against a real registry until M4 lands... the caller... is responsible for
producing that resolved input once the registry exists"), M4_ONBOARDING §6
(the actual `RegistryResult` contract: `{status: "in_force"|"repealed"|
"superseded"|"not_found", as_of, note}`).

**Current implementation (before fix).** `eval/run_eval.py::score_repealed`
called `item.as_registry_result()`, which reinterpreted the gold set's own
`in_force_label` as if it were a real registry's answer, then fed that into
`citation_validity_rate`/`repealed_citation_rate`. For the 34-item seed set
(26 `repealed` / 8 `in_force` by construction), this produced exactly `0.7647`
(= 26/34) as a "measured" `repealed_citation_rate` — again, dataset composition
presented as a system result. Separately, and independently of the fabrication
issue, the *shape* my `as_registry_result()` invented
(`resolved/exists/in_force/superseded_by`) does not match the real
`RegistryResult` M4 will actually produce (`status/as_of/note`) — so even a
correct future caller would need an unplanned translation layer, contradicting
the onboarding's explicit intent.

**What's wrong.** Two independent defects: (a) a fabricated top-line number in
the live run, same category as G1; (b) an interface designed against an
invented contract instead of the one already specified in `M4_ONBOARDING.md`,
which is exactly the kind of "different architecture than the project expects"
Step 4 of this audit was asked to hunt for.

**Corrective action.**
1. Redefine `citation_validity_rate` / `repealed_citation_rate` in
   `eval/metrics.py` to accept M4's real shape
   (`{status, as_of, note}`), and add a `registry_coverage_rate` metric
   (RESEARCH.md T6: "report coverage rate... out-of-registry citations flagged,
   not silently passed" — a distinct, explicitly named requirement that had no
   corresponding metric at all before this audit).
2. Update `RepealedItem.as_registry_result()` to emit the corrected shape, so
   it remains useful as a metric-testing fixture without also being fed into a
   live run as if it were real.
3. In `run_eval.py::score_repealed`, stop calling `as_registry_result()` against
   the live run entirely. Report `citation_validity_rate`,
   `repealed_citation_rate`, and `registry_coverage_rate` as `unavailable` with
   reason `"no_registry_available_M4_not_started"` — honest, and consistent with
   the onboarding's own statement that these "can't be computed... until M4
   lands."

**Files affected.** `eval/metrics.py` (`citation_validity_rate`,
`repealed_citation_rate`, new `registry_coverage_rate`), `eval/schemas.py`
(`RepealedItem.as_registry_result`), `eval/run_eval.py` (`score_repealed`),
`tests/test_metrics.py`, `tests/test_eval_schemas.py`, `tests/test_run_eval.py`.

---

## HIGH

*(No additional HIGH-severity gaps beyond G1/G2, which were assessed CRITICAL
because they produce a fabricated number in the primary deliverable — the
results report — rather than merely an incomplete feature.)*

---

## MEDIUM

### G3 — `scipy` listed as an M6 dependency but never actually used

**Source requirement.** M6_ONBOARDING §14 (`[eval]` extras include
`"scipy>=1.11   # bootstrap resampling / ECE binning"`), RESEARCH.md §5 ("never
hand-rolled where a real library exists").

**Current implementation (before fix).** `pyproject.toml` correctly lists
`scipy>=1.11` in `[eval]`. Nothing in `eval/` imports it. `eval/gold_sets.py`'s
`proportion_ci_halfwidth` and `eval/stats.py`'s `proportion_ci` each hand-roll a
`{0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}` lookup instead of using
`scipy.stats.norm.ppf`, which is exactly the hand-rolled-primitive pattern
RESEARCH.md §5 objects to elsewhere (ROUGE, BERTScore).

**Corrective action.** Replace both hardcoded lookups with
`scipy.stats.norm.ppf`, making the declared dependency load-bearing.

**Files affected.** `eval/gold_sets.py`, `eval/stats.py`.

### G3b — CI regression gate false-positives on latency jitter (found while verifying G1–G4's fixes)

**Source requirement.** M6_ONBOARDING §11: the CI gate should guard "against
the harness silently breaking, not against real quality regressions."

**How it was found.** Not from re-reading a document — from re-running the
regression gate twice in a row against the corrected baseline as part of
verifying G1/G2, and watching it fire on `latency_p50_ms[retrieve]` and
`latency_p95_ms[retrieve]` purely from ordinary wall-clock jitter between two
back-to-back runs of identical gold data (0.78ms → 0.62ms, a real but
meaningless difference).

**What's wrong.** `compare_to_baseline` compared every "ok" metric, including
timing, against one committed baseline with one absolute tolerance. Timing
naturally varies run to run on the same machine; a gate that fires on that
basis trains reviewers to ignore it, defeating its purpose.

**Corrective action.** Excluded `latency_*` metrics from the regression
comparison (`eval/report.py::_is_quality_metric`). Latency is still computed
and reported in every results file — only the pass/fail regression check
ignores it.

**Files affected.** `eval/report.py`, `tests/test_report.py`.

---

### G4 — Results manifest does not record `split_version`

**Source requirement.** OPS.md §4: "Every `RunManifest` and every results file
records `split_version`."

**Current implementation (before fix).** `eval/run_eval.py`'s manifest has
`"split": "dev"` (the split *name*) but no `split_version` field at all.

**Corrective action.** Add `split_version` to the manifest, read from
`data/splits/VERSION` (or similar) when it exists, else explicitly `null` with
a `split_version_note` explaining that `data/splits/*.json` has not been frozen
yet (M1 not landed) — matching the project's established convention of an
honest, stated absence rather than a silently missing field.

**Files affected.** `eval/run_eval.py`.

---

## LOW

### G5 — Ordering deviation from `M6_ONBOARDING.md` §17's literal step sequence

The onboarding lists the annotation app as step 4, before the schema/gold-set
infrastructure. The actual build order here put schemas and storage first,
because the app cannot write to a store it doesn't know the shape of. This was
disclosed explicitly in `docs/M6_IMPLEMENTATION_PLAN.md` Phase 1 at the time.
No code change warranted; recorded here only so the audit's own record shows
this was checked, not missed.

### G6 — `eval/config.py` is an inferred file, not a literally named deliverable

CLAUDE.md §4's "no magic numbers outside `config.py`" names one file, under
`src/smartlawai/`, for serving thresholds. `eval/config.py` generalises that
convention to the eval package; no M6 document names this file explicitly, but
none contradicts it either, and centralising ~20 M6-specific constants
(seeds, bootstrap resamples, IAA rates, target Ns) is a reasonable application
of an existing convention rather than an invented one. No change made.

### G7 — Reference images could not be re-verified in this audit turn

Recorded as a limitation in the traceability matrix header. The original
transcription (`docs/M6_IMPLEMENTATION_ANALYSIS.md` §9) is the only available
record; nothing in this audit could contradict or confirm it against fresh
pixels. No requirement was found to hinge critically on a detail only visible
in the images and absent from the text docs, so this is recorded as a process
limitation, not a live gap.

---

## Unsupported Assumptions or Deviations (Step 4)

Per the audit's explicit instruction to hunt for hallucinated implementation:

1. **`eval/build_seed_gold.py` and the seed gold sets themselves.** Not
   mentioned by name in any source document. Justification: `M6_ONBOARDING.md`
   §2's done-when criterion requires gold sets to exist for `run_eval.py` to
   run against, and a real annotation study needs live human raters this audit
   cannot supply. The seed sets are the minimum honest way to satisfy the
   structural requirement without a live study — and every row is labelled
   with `n_annotations=1`, `agreement=None`, and a `provenance` string stating
   exactly what it is. **Retain.** The alternative (leaving `eval/gold/`
   empty) would make the done-when criterion permanently unverifiable by
   anyone without first running a multi-day human study, which is a worse
   outcome for a milestone whose own spec says "the full table computes... is
   your floor."
2. **The two `score_groundedness`/`score_repealed` fabrication bugs (G1, G2).**
   These were not deliberate scope additions — they were an implementation
   shortcut taken under generic "make the metric produce *a* number" instinct,
   which is precisely the kind of "used a different architecture than the
   project expects" and "used placeholder logic instead of required logic"
   failure mode Step 4 asks to name. **Corrected**, not retained.
3. **`RegistryResult`'s invented shape (part of G2).** Built without checking
   `M4_ONBOARDING.md`'s already-published contract first. **Corrected.**
4. **`eval/README.md` and `data/perturbations/README.md`.** Pure documentation,
   not named in any source file, added for a future reader's convenience.
   **Retain** — no functional risk, no scope conflict, nothing they claim
   contradicts the source docs.
5. **Everything else inspected during this audit** (schemas, gold-set loader,
   annotation store, agreement statistics, pooling, stats, extractiveness, the
   perturbation generator, the CI workflow) traces to an explicit requirement
   and was not found to conflict with any source document, M0–M5 interface, or
   the images' transcription. No further deviations found.
