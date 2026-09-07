# M6 Compliance Report

Produced by an independent audit of the M6 implementation against the project's
own source materials, treating the prior completion report as untrusted rather
than as evidence. See `docs/M6_REQUIREMENTS_TRACEABILITY.md` (the full
requirement-by-requirement matrix) and `docs/M6_GAP_ANALYSIS.md` (every gap
found, with corrective action and file-level evidence) for the detail behind
every claim in this report.

## 1. Source materials reviewed

Re-read in full during this audit (not from memory of the earlier session):

- `CLAUDE.md` (158 lines, full)
- `PLAN.md` (426 lines, full — not only §M6)
- `RESEARCH.md` (539 lines, full — §1 through §12)
- `OPS.md` (539 lines, full — §1 through §10)
- `M6_ONBOARDING.md` (614 lines, full — §1 through §18)
- `M1_ONBOARDING.md` §7–9 (dedup/split, provenance, leakage tests)
- `M4_ONBOARDING.md` §6 (the `RegistryResult` contract)

**Limitation, disclosed rather than hidden**: the two M6 reference images could
not be re-opened in this audit turn (no image was re-attached to the
conversation). Rows in the traceability matrix sourced from them cite the
transcription made during the original viewing
(`docs/M6_IMPLEMENTATION_ANALYSIS.md` §9), not a fresh re-read. No requirement
was found to hinge on an image detail absent from the text documents.

## 2. Requirements summary

M6 builds the scoring system the whole project is judged by, plus the
human-labelled test sets scoring depends on: `eval/metrics.py`/`stats.py`
(standard libraries, never hand-rolled), the annotation app (built first, per
`PLAN.md`), five gold sets with per-set IAA duplication rates, the perturbation
generator for M8a's judge, and `eval/run_eval.py` producing a complete results
table for the untrained (stubbed) system — the literal done-when floor.

## 3. Compliance score

Counted from the 38-row traceability matrix:

| Category | Count |
|---|---|
| Total requirements identified | 38 |
| Fully implemented (after this audit's corrections) | 30 |
| Partially implemented (genuine, disclosed limitations) | 4 |
| Not started / blocked (human-annotator-time limitations) | 3 |
| Not applicable to M6 | 1 |

**Six requirements were IMPLEMENTED BUT INCORRECT before this audit** (R4, R5,
R21, R25, R29, R33 in the traceability matrix) — all six traced to the same two
root-cause defects (G1, G2 in the gap analysis) and are corrected as of this
report. One further defect (G3b, a false-positive CI regression trigger on
timing jitter) was found *while verifying* those corrections and is also fixed.

## 4. Requirement-by-requirement status

See `docs/M6_REQUIREMENTS_TRACEABILITY.md` for the full 38-row matrix with
file-level evidence per row. Summary of what changed in this audit pass:

| Requirement | Before this audit | After this audit |
|---|---|---|
| Claim-level groundedness (R4, R5, R29) | Computed from the gold label itself — tautological, always exactly reproduced the dataset's own 50/50 label split | Computed from the pipeline's real `Verifier` component; a separate, correctly-named judge-agreement metric (Cohen's κ + accuracy vs human labels) added |
| Citation validity / repealed-citation rate (R5, R21, R33) | Computed by feeding the gold label back in as a fake registry answer, in an invented shape that didn't match M4's real interface | Reports `unavailable: no_registry_available_M4_not_started`; the accepted input shape now matches `M4_ONBOARDING.md` §6's real `RegistryResult` exactly |
| Registry coverage rate (R34) | Did not exist | Added — RESEARCH.md T6 names this explicitly |
| scipy dependency (R11, R12) | Declared, never imported; two hand-rolled z-value tables | `scipy.stats.norm.ppf` now backs both CI calculations, lazily imported so base functionality survives without the `[eval]` extra |
| `split_version` in results manifest (R30) | Field absent | Present, explicitly `null` with a stated reason until M1 freezes `data/splits/*.json` |
| CI regression gate on latency (found during verification) | Would false-positive on ordinary wall-clock jitter | Excludes timing metrics from the pass/fail check; still reports them |

Every other row in the matrix was checked and found to already match its
source requirement; see the matrix for the specific evidence per row.

## 5. Corrections made

1. **`eval/metrics.py`** — `citation_validity_rate`, `repealed_citation_rate`
   rewritten to accept M4's real `RegistryResult` shape (`status`/`as_of`/
   `note`); new `registry_coverage_rate` added.
2. **`eval/schemas.py`** — `RepealedItem.as_registry_result()` rewritten to
   emit the corrected shape.
3. **`eval/run_eval.py`** — `score_groundedness` rewritten to call
   `corpus.pipeline.verifier.verify()` on each gold `(claim, passage)` pair
   instead of reading the gold label; adds `judge_agreement_kappa` and
   `judge_accuracy_vs_human`; drops the erroneous `citation_coverage` call
   (measured an unpopulated gold field, not system output). `score_repealed`
   rewritten to report `unavailable` with a stated reason instead of computing
   a number from `as_registry_result()`. New `_split_version()` helper; new
   `split_version`/`split_version_note` manifest fields.
4. **`eval/gold_sets.py`, `eval/stats.py`** — `proportion_ci_halfwidth` and
   `proportion_ci` now use `scipy.stats.norm.ppf` (lazy import); `describe_gold_set`
   and `sizing_note` degrade gracefully to an explicit "not computed" if scipy
   is unavailable, rather than crashing a whole run.
5. **`eval/report.py`** — `compare_to_baseline` now excludes `latency_*`
   metrics from the regression check via a new `_is_quality_metric` filter.
6. **Tests updated to match**: `tests/test_metrics.py` (registry-metric fixtures
   and assertions rewritten for the new shape), `tests/test_eval_schemas.py`
   (`as_registry_result` tests rewritten), `tests/test_run_eval.py` (five tests
   rewritten to assert verifier-driven behaviour instead of label-echo; two new
   tests added for the judge-agreement metrics and the removed citation-coverage
   call), `tests/test_eval_integration.py` (one call site updated), `tests/test_report.py`
   (one new test for the latency-jitter fix).
7. **Committed baseline regenerated** — `eval/results/dev_20260905T165639Z.{json,md}`
   replaces the earlier (fabricated-looking) baseline; the corrected numbers
   are visibly more honest (e.g. `groundedness.judge_agreement_kappa: 0.0000`,
   correctly showing the current stub verifier does no real judging, versus the
   earlier `groundedness_rate: 0.5000` that was actually just the dataset's own
   composition).
8. **New audit documents**: `docs/M6_REQUIREMENTS_TRACEABILITY.md`,
   `docs/M6_GAP_ANALYSIS.md`, this report.

## 6. Remaining limitations

Only limitations genuinely supported by the current project state:

1. **The five gold sets are seeds (10–40 items, single-labelled), not the
   300–400-item multi-rater study `RESEARCH.md` §7 specifies.** This requires
   live human annotators over multiple sessions; no AI-agent session can
   produce it. Disclosed per-row in `eval/gold/PROVENANCE.md` and in the
   traceability matrix (R17) as PARTIALLY IMPLEMENTED, not IMPLEMENTED.
2. **The 20-item pilot required before scaling any set (`RESEARCH.md` §7.3)
   has not run** — same root cause (R16, BLOCKED). The machinery to run and
   measure it (`AnnotationStore.throughput()`, `.agreement()`) is built and
   tested; no session has occurred.
3. **`retrieval.jsonl`'s judgment pools are empty by necessity**, not
   oversight — `doc_id`/`chunk_id` are fresh UUIDs on every ingest
   (`src/smartlawai/core/ocr.py:53`, `core/chunking.py:52`), so a committed
   pool would reference nothing on the next run. Becomes actionable once M2
   gives the corpus stable ids; the TREC-pooling code is ready.
4. **ClaimRAG-Law (968 validated claims) is not publicly released** — checked
   directly against the arXiv abstract, the full HTML (v4), and a web search;
   the release links were omitted for double-blind review and no repository
   was found. Documented per `M6_ONBOARDING.md` §10's explicit instruction to
   state this rather than substitute something else.
5. **`repealed.jsonl`'s successor mappings are built from the replacing Acts'
   own section text, not verified row-by-row against the Gazette of India.**
   Flagged in `authority_source` on every row and in `eval/gold/PROVENANCE.md`.
6. **`bert-score` is not installed in this environment**; `bertscore_f1` is
   untouched and its test skips honestly with `pytest.importorskip`.
7. **A pre-existing, unrelated flaky test** —
   `tests/test_local_backend.py::test_faiss_lifecycle` (unseeded
   `np.random.rand`, fails ~60% of the time) — predates this work entirely
   (confirmed via `git diff`, which shows this session touched only one
   unrelated line in that file) and is outside M6's scope.

No limitation from the earlier completion report was found to be
understated — if anything, the earlier report's framing of the gold-set gap
was accurate; the errors this audit found were in the *metric computation*
(G1/G2), not in what was already disclosed about the gold sets themselves.

## 7. Test results

Full repository suite, after every correction in this audit:

```
397 passed, 2 skipped, 1 deselected in ~10s
```

- **1 deselected**: `tests/test_local_backend.py::test_faiss_lifecycle` — the
  pre-existing, unrelated flaky test (item 7 above); deselected only for this
  deterministic count, not disabled in the repository.
- **2 skipped, both honest**: `bert-score` not installed (stated reason);
  `data/splits/train.json` not landed yet (stated reason, M1's territory).

M6-specific test files (the ones this milestone owns or extended):

```
tests/test_eval_schemas.py, test_gold_sets.py, test_stats.py,
test_extractiveness.py, test_agreement.py, test_annotation_store.py,
test_pooling.py, test_perturbations.py, test_run_eval.py, test_report.py,
test_eval_integration.py, test_metrics.py
  → 348+ passed (exact count shifts slightly with the tests added/rewritten
    in this audit; verified passing as part of the full-suite run above)
```

Integration / end-to-end:

```
python -m eval.run_eval --split dev
  → split=dev items=42 metrics_ok=31 metrics_unavailable=17 failures=0
  → wrote eval/results/dev_20260905T165639Z.json
  → wrote eval/results/dev_20260905T165639Z.md

python -m eval.run_eval --split dev --baseline eval/results/dev_20260905T165639Z.json
  → [regression] OK: no metric regressed past 0.1
```

Layering (`OPS.md` §8, statically enforced):

```
tests/test_eval_integration.py::test_metric_layer_does_not_import_smartlawai   PASS
tests/test_eval_integration.py::test_only_run_eval_imports_the_pipeline_class  PASS
```

`ruff check --select E4,E7,E9,F --line-length 100` (the project's actual
configured rule set, per `pyproject.toml`'s `[tool.ruff]`): clean on every file
this milestone created or touched, before and after the audit's corrections.

Baseline for comparison: **61 tests passed** before any M6 code existed
(measured at the start of the original implementation session). No pre-existing
test has been broken at any point, including through this audit.

## 8. Final verdict

**MOSTLY COMPLIANT — MINOR GAPS REMAIN**

Justification for this verdict rather than "fully compliant": two CRITICAL
defects existed in the results the milestone produces (fabricated-looking
groundedness and repealed-citation numbers) until this audit found and fixed
them — a "fully compliant" verdict cannot be honestly claimed for a
self-audited session where the primary deliverable needed a correction of that
severity, even though the correction is now complete and verified. The
remaining gaps (the gold sets being seeds rather than a completed human study,
and the unrun pilot) are structural limitations of what a single AI-agent
session can deliver without live human annotators, are explicitly disclosed
everywhere they matter, and do not represent incomplete engineering — the
machinery for all of it is built and tested. Nothing in the current
implementation contradicts a source document, uses a wrong architecture, or
hides a known defect.
