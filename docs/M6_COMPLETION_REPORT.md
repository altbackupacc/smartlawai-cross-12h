# M6 Completion Report — Evaluation Harness + Gold Sets

> **Superseded in part by a later independent audit.** After this report was
> written, a strict requirements audit
> (`docs/M6_REQUIREMENTS_TRACEABILITY.md`, `docs/M6_GAP_ANALYSIS.md`,
> `docs/M6_COMPLIANCE_REPORT.md`) found that two sections of the results this
> milestone produces — claim-level groundedness and citation/repealed-citation
> rates — were computed by reading the gold set's own label back as if it were
> a system measurement, which is a fabricated-metric bug (CLAUDE.md I2) despite
> looking like a normal result. **Both were corrected** (see the compliance
> report §5 for the fix detail); the specific numbers quoted below (e.g.
> `groundedness_rate: 0.5000`, `repealed_citation_rate: 0.7647`, and the
> `dev_20260905T162402Z` results filename) are the **pre-correction** values
> and no longer match the repository — the committed baseline is now
> `dev_20260905T165639Z`. Read `docs/M6_COMPLIANCE_REPORT.md` for
> the current, audited state; this file is kept for its still-accurate
> narrative of what was built and why.

Companion documents: `docs/M6_IMPLEMENTATION_ANALYSIS.md` (what existed, what was
missing) and `docs/M6_IMPLEMENTATION_PLAN.md` (the 14-phase plan this report
tracks against). This report states what was actually built, what was
deliberately left as a seed rather than a finished study, and how to run
everything.

---

## 1. What was implemented

All 14 planned phases, in full:

1. Architecture, `eval/` package skeleton, `eval/config.py`, and the §4 move
   of `src/smartlawai/eval/metrics.py` → `eval/metrics.py`.
2. Typed schemas for all five gold sets plus the annotation record
   (`eval/schemas.py`).
3. Gold-set loading, validation, checksumming, and the stated power calculation
   (`eval/gold_sets.py`).
4. Annotation rubrics as versioned data and an append-only annotation store
   with export straight to `eval/gold/*.jsonl` (`eval/annotate/guidelines.py`,
   `eval/annotate/store.py`).
5. The evaluation runner core (`eval/run_eval.py`).
6. Retrieval metrics: Recall@{5,10,20}, Precision@{5,10,20}, MRR, nDCG@10.
7. Answer-quality metrics: `rouge-score`-backed ROUGE-1/2/L (hand-rolled
   `rouge_l` deleted), extractiveness (`eval/extractiveness.py`), claim-level
   groundedness/hallucination rate.
8. Citation metrics: coverage, validity rate, repealed-citation rate,
   citation-support rate — all accepting pre-resolved registry results, with no
   import of M4's non-existent `registry/`.
9. Robustness: refusal precision/recall/F1/over-refusal, per-perturbation-type
   detection rate, and the seven-type synthetic perturbation generator
   (`data/perturbations/generate_perturbations.py`).
10. The Streamlit annotation app (`eval/annotate/app.py`) and TREC pooling
    (`eval/annotate/pooling.py`).
11. Statistical protocol: paired bootstrap, ECE, reliability-diagram data,
    Bonferroni correction, mean±std, Wilson CI (`eval/stats.py`).
12. Reporting: JSON + human-readable Markdown, with an explicit "not measured"
    section (`eval/report.py`).
13. End-to-end integration: `run_eval.py` against the real (stubbed) `Pipeline`,
    a committed dev baseline, and a CI regression-gate workflow.
14. Testing, ruff, and this report.

## 2. Final M6 architecture

```
                    SmartLawAI Pipeline (unmodified, I1/I3 intact)
                                   |  Pipeline.ask(question, Scope)
                                   v
 eval/gold/*.jsonl --> eval/gold_sets.py --> eval/run_eval.py --> eval/results/*.json
      ^  (frozen)      (schema-validated)         |                        + *.md report
      |                                           |
 eval/annotate/app.py                             +--> eval/metrics.py    (retrieval, answer,
   |- guidelines.py  (rubrics as data)            |    eval/extractiveness.py  citation, refusal,
   |- pooling.py     (TREC pooling)               |                            perturbation)
   |- store.py       (append-only JSONL, export)  +--> eval/stats.py      (bootstrap, ECE,
   +- agreement.py   (kappa / alpha, anchoring)   |                        Bonferroni, power)
                                                  +--> eval/report.py     (Markdown table)

 data/perturbations/generate_perturbations.py --> perturbations.jsonl (M8a training data)
      source: data/raw/.../IN-Abs (train-data)   guard: tests/test_perturbations.py
```

**Layering, enforced by `tests/test_eval_integration.py`:** every module in
`eval/` except `run_eval.py` imports nothing from `smartlawai`. `run_eval.py` is
the sole, documented exception — it calls `Pipeline.ask()` unmodified, exactly
as `M6_ONBOARDING.md` §11–12 requires, and nowhere else in `eval/` reimplements
any part of the retrieve/rerank/generate/verify/decide sequence.

## 3. Files created

```
eval/__init__.py, config.py, schemas.py, gold_sets.py, metrics.py (moved),
     stats.py, extractiveness.py, report.py, run_eval.py, build_seed_gold.py,
     README.md
eval/annotate/__init__.py, guidelines.py, store.py, agreement.py, pooling.py, app.py
eval/gold/{retrieval,oos,groundedness,repealed,entailment}.jsonl, PROVENANCE.md
eval/gold/external/PROVENANCE.md
eval/results/dev_20260905T165639Z.{json,md}   (the committed CI baseline)
data/perturbations/generate_perturbations.py, README.md
.github/workflows/eval-regression.yml
tests/test_eval_schemas.py, test_gold_sets.py, test_stats.py,
      test_extractiveness.py, test_agreement.py, test_annotation_store.py,
      test_pooling.py, test_perturbations.py, test_run_eval.py, test_report.py,
      test_eval_integration.py
docs/M6_IMPLEMENTATION_ANALYSIS.md, M6_IMPLEMENTATION_PLAN.md,
     M6_COMPLETION_REPORT.md (this file)
```

## 4. Files modified

| File | Change |
|---|---|
| `tests/test_metrics.py` | Import path → `eval.metrics`; ROUGE tests retargeted at `rouge_scores`; extended with ~45 new cases for the metrics added in Phases 6–9 |
| `tests/test_local_backend.py` | One import line: `smartlawai.eval.metrics` → `eval.metrics`, `rouge_l` → `rouge_scores` |
| `conftest.py` | Added the repo root to `sys.path` so `import eval.metrics` resolves from tests |
| `pyproject.toml` | Added the `[eval]` extras group (`rouge-score`, `bert-score`, `scipy`) |
| `.gitignore` | Added `data/perturbations/perturbations.jsonl` and `eval/annotations/` |

`src/smartlawai/eval/` (both `__init__.py` and the old `metrics.py`) was
**deleted** after the move — no duplicate left behind, per
`M6_ONBOARDING.md` §4.3.

**Nothing else was touched.** `pipeline.py`, `scope.py`, `trace.py`,
`protocols.py`, `config.py`, `adapters/*`, `core/*`, `guardrails/*`,
`api/main.py`, `ui/app.py`, `cli.py`, `train/**`, `gcloud/**`, `docker/**`, the
five pre-existing `.github/workflows/*.yml`, and everything under `data/`
except the new `data/perturbations/` are unmodified.

## 5. Existing components reused

- `Pipeline.ask(question, scope)` / `Pipeline.ingest(...)` — called unmodified.
- `Scope`, `PipelineTrace`, `Claim`, `GenerationResult`, `VerificationResult`.
- All nine keeper functions from the original `metrics.py` (`token_f1`, `prf`,
  `bertscore_f1`, `ner_f1`, `ner_per_label_f1`, `mean_reciprocal_rank`,
  `recall_at_k`, `precision_at_k`, `citation_recall`, `citation_precision`) —
  moved, not rewritten.
- `sample_docs/sample_judgment.txt` — structural smoke input only.
- `data/raw/in_ext_source/dataset/IN-Abs/{train,test}-data` — perturbation
  source and seed-gold-set source respectively; not re-pulled.
- `pyproject.toml`'s existing `[ui] streamlit>=1.30`.
- M1's `pytest.skip("... not landed")` convention, applied to the perturbation
  generator's train-split-only check.

## 6. Metrics implemented

Full `RESEARCH.md` §5 table, in `eval/metrics.py` / `eval/extractiveness.py` /
`eval/stats.py`:

| Family | Metrics |
|---|---|
| Retrieval | Recall@{5,10,20}, Precision@{5,10,20}, MRR, nDCG@10 |
| Summarisation | ROUGE-1/2/L (`rouge-score`), BERTScore (`bert-score`, present but not installed — see §11), Extractiveness (4-gram) |
| QA / groundedness | `groundedness_rate`, `hallucination_rate` (judge-based, exact complements — CLAUDE.md §5's single definition); `judge_agreement_kappa`, `judge_accuracy_vs_human` (agreement with human gold labels) |
| Citation | `citation_coverage`, `citation_support_rate` (computable today); `registry_coverage_rate`, `citation_validity_rate`, `repealed_citation_rate` (functions implemented, correctly report `unavailable` in a live run until M4's registry exists — see the audit's compliance report) |
| Refusal / OOS | `refusal_precision/recall/f1`, `over_refusal_rate`, `refusal_accuracy` |
| Judge diagnostics | `per_perturbation_detection_rate` (per type) |
| Calibration | `expected_calibration_error`, `reliability_diagram_data` |
| Cost/latency | `latency_p50_ms` / `latency_p95_ms` per stage, from `PipelineTrace` |
| Statistics | `paired_bootstrap` (10k resamples), `bonferroni_correct`, `mean_std`, `proportion_ci` (Wilson) |

Every metric returns `None` with a stated reason when it cannot be computed
(CLAUDE.md I2) — never a substituted number. `eval/report.py` renders this as
`unavailable` and lists every such case in a dedicated "Not measured" section.

## 7. Test sets supported

All five named in `PLAN.md`/`RESEARCH.md`: `retrieval.jsonl`, `oos.jsonl`,
`groundedness.jsonl`, `repealed.jsonl`, `entailment.jsonl`. See §9 below for
their current honest state.

`eval/gold/external/` is empty by design — see §10.

## 8. How to run M6

```bash
pip install -e ".[eval,ui,dev]"

# Run the harness over every gold set the repo ships:
python -m eval.run_eval --split dev

# Compare against a committed baseline (what CI does on every PR):
python -m eval.run_eval --split dev --baseline eval/results/dev_20260905T165639Z.json

# Rebuild the seed gold sets:
python -m eval.build_seed_gold

# Regenerate the perturbation training data (M8a's input, not eval data):
python data/perturbations/generate_perturbations.py --n 500
```

## 9. How to add new test cases

Programmatically, using the schemas directly (`eval/schemas.py`):

```python
from eval.schemas import OOSItem
from eval.gold_sets import load_gold_set, write_gold_set

items = load_gold_set("oos")
items.append(OOSItem(item_id="oos-new-0001", question="...", label="in_scope",
                     provenance="how this label was produced"))
write_gold_set("oos", items)
```

Through the app (the intended path for real study data): run
`streamlit run eval/annotate/app.py`, point it at a queue file of unlabelled
items, and export.

## 10. How to run human evaluation

```bash
streamlit run eval/annotate/app.py
```

1. Choose a gold set and enter a rater id (any opaque string — the schema and
   UI never distinguish rater pools, per `PLAN.md` M6).
2. The app shows one blinded item per screen in a per-rater deterministic
   random order; ~10% are cold-mode (no pre-filled label) for the
   `RESEARCH.md` §7.1.1 anchoring control.
3. Submit; the sidebar shows live Cohen's κ / Krippendorff's α and
   items-per-hour throughput as data comes in.
4. Click "Export to gold set" to write straight to `eval/gold/<task>.jsonl` —
   no spreadsheet step.

`eval/annotate/agreement.py` and `eval/annotate/pooling.py` are also usable
standalone for reporting a completed study (`RESEARCH.md` §7.5) or for
preparing a TREC-pooled retrieval queue.

## 11. How to generate reports

`python -m eval.run_eval` always writes both a JSON and a Markdown report to
`eval/results/`. To render a report from an already-computed results dict
elsewhere:

```python
from eval.report import write_report
write_report(results, out_dir, stem="my_run")
```

## 12. Test results

```
389 passed, 2 skipped in ~10s   (full repo suite, one unrelated pre-existing
                                  flaky test deselected -- see §13)
348 passed, 2 skipped            (M6 test files + tests/test_metrics.py only)
```

Baseline before this work: **61 passed** (verified by running `pytest -q`
before any M6 code was written). No pre-existing test was broken.

`ruff check --select E4,E7,E9,F` (the project's actual configured rule set,
confirmed from `pyproject.toml`'s `[tool.ruff]`, which sets only
`line-length = 100`): every file M6 created or modified is clean. Four
pre-existing `F541` findings in `data/score_filter_pairs.py` (M1's file, not
touched) are unrelated and left as found.

The two honest skips:
- `test_bertscore_is_present_but_unexercised` — `bert-score` is not installed
  in this environment (see §11 below for the finding this closes out).
- `test_perturbation_sources_are_drawn_from_the_training_split` —
  `data/splits/train.json` does not exist yet (M1 has not frozen the splits);
  skips with that reason stated, becomes a real assertion with zero code
  changes the moment M1 lands it.

**The M6 done-when criterion is met:** `python -m eval.run_eval --split dev`
runs the real (stubbed) `Pipeline` over all five shipped gold sets end to end,
computes 31 metrics and reports 14 as honestly `unavailable`/`not_applicable`
with stated reasons, raises zero failures, and writes a committed results table
(`eval/results/dev_20260905T165639Z.{json,md}`).

## 13. Known limitations

State these plainly; none were papered over.

1. **The gold sets are seeds, not the study.** `RESEARCH.md` §7 specifies
   300–400 items per set collected by multiple raters with a reported agreement
   statistic. What ships is 10–40 items per set, single-labelled, with
   `agreement: null` and a `provenance` string on every row. `eval/gold/PROVENANCE.md`
   states this in detail, including the per-set achieved-*n* CI half-width. No
   agreement number is invented anywhere.
2. **`retrieval.jsonl` has empty judgment pools.** `doc_id`/`chunk_id` are fresh
   UUIDs on every ingest (`src/smartlawai/core/ocr.py:53`,
   `core/chunking.py:52`), so committed chunk ids would be meaningless on the
   next run. Retrieval metrics correctly report `unavailable` rather than a
   fabricated score. This becomes actionable once M2 gives the corpus stable
   ids; the pooling and annotation machinery (`eval/annotate/pooling.py`) is
   ready today.
3. **`repealed.jsonl`'s successor mappings are unverified against primary
   sources.** They were built from the replacing Acts' own section text, not
   checked row-by-row against the Gazette of India. Flagged per-row in
   `authority_source` and in `eval/gold/PROVENANCE.md`; do not publish a number
   from this set before that verification.
4. **ClaimRAG-Law is not publicly released.** Checked the arXiv abstract, the
   full HTML (v4), and a web search; the paper states the dataset "will be
   publicly available" but the release links were omitted for double-blind
   review and no repository was found anywhere. Documented explicitly in
   `eval/gold/external/PROVENANCE.md` per `M6_ONBOARDING.md` §10's instruction
   to say so rather than substitute something silently. This blocks the
   external generalisation check in `RESEARCH.md` §2.5/T11 until either the
   camera-ready version restores the link or the authors are contacted
   directly.
5. **Claim-level groundedness cannot be scored from live pipeline output yet.**
   `AnswerResult` exposes claim counts, not claim texts (`trace.generation`
   only carries `n_claims`). `run_eval.py` scores groundedness from the gold
   `(passage, claim)` pairs instead — exactly what `groundedness.jsonl` is for
   — without widening `pipeline.py`'s public API, which M6 must not touch. When
   M3/M5 land, the same function scores real claims unchanged, because the gold
   schema already mirrors `protocols.Claim`'s field names.
6. **Calibration (ECE) is `unavailable` on the stub pipeline by design, not by
   gap.** The obvious substitute — scoring the gate's own ACCEPT/REFUSE outcome
   as ground truth — is circular (the gate refuses exactly when confidence is
   low, so confidence would predict itself and every system would look
   perfectly calibrated). `calibration_metrics()` requires an independent
   `correctness_by_item` gold label and reports `unavailable` without one; this
   is intentional and tested (`tests/test_run_eval.py`).
7. **`bert-score` is not installed in this repository.** No test ever exercised
   `bertscore_f1` before this work (confirmed by grep); it still is not
   exercised now, but the gap is explicit: `tests/test_metrics.py` skips it with
   `pytest.importorskip` and states the reason, rather than silently passing or
   silently dropping the metric.
8. **A pre-existing, unrelated flaky test was found during verification, not
   caused by this work:** `tests/test_local_backend.py::test_faiss_lifecycle`
   uses unseeded `np.random.rand` and fails roughly 60% of the time
   independent of any M6 change (confirmed by running it in isolation five
   times, and by `git diff` showing this work touched only one unrelated line
   in that file). Left as found — it is M0 territory, not M6's, and fixing it
   was outside this milestone's scope.
9. **The environment's `.venv` needed a local repair** (its `pyvenv.cfg`
   pointed at a different machine's Python install and could not start).
   Repointed at the local Python 3.11 interpreter; the original is preserved as
   `.venv/pyvenv.cfg.bak`. `.venv/` is gitignored, so this is a local dev-machine
   fix, not a repository change, and does not appear in the diff.
