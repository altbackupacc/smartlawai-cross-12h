# M6 Implementation Analysis — Evaluation Harness + Gold Sets

Written before any M6 code was added. Sources of truth, in precedence order:
`CLAUDE.md` → `PLAN.md` §M6 → `RESEARCH.md` §5–§7 → `M6_ONBOARDING.md` → the two
M6 reference images. Where the images and the prose disagree, the prose wins; in
practice they agree (§9 records the mapping).

---

## 1. Current SmartLawAI architecture

The repo is at **M0 complete / M1 mid-flight**. Everything below was verified by
reading the files, not inferred.

```
src/smartlawai/
├── config.py         all thresholds + stub model ids (CLAUDE.md §4: no magic numbers elsewhere)
├── scope.py          Scope(doc_ids, owner_id) frozen dataclass — I1
├── trace.py          PipelineTrace / StageRecord + .stage() context manager, .to_dict()
├── protocols.py      Encoder/Reranker/Generator/Verifier Protocols
│                     + Claim, GenerationResult, VerificationResult dataclasses
├── pipeline.py       THE pipeline: ingest() and ask(question, scope) -> AnswerResult
├── adapters/         base.py (BackendInterface ABC + DTOs), local.py (DuckDB), gcloud.py, factory.py
├── core/             ocr, preprocess, chunking, bm25, rerank, rag, ner, clauses, summarize, ...
├── guardrails/       pii.py, disclaimer.py
└── eval/             __init__.py + metrics.py   <- the misplaced module M6 owns (section 4 below)
api/main.py           FastAPI, thin handlers
ui/app.py             Streamlit M0 skeleton (ingest + ask)
train/                common/{device,seeding,checkpointing}, encoders/, qlora/
data/                 M1 territory: prepare_iltur.py, align_rr_summ.py, build_pairs.py, raw/, pilot/
tests/                8 files, 61 tests, all green (verified: `pytest -q` -> 61 passed in 129s)
```

There is **no top-level `eval/` directory yet**. M1's `eval/ocr_wer.py` has not landed,
so M6 creates the directory.

**Environment note (found during analysis, fixed):** the shipped `.venv/` was created on
a different machine (`pyvenv.cfg` pointed at `C:\Users\kasir\...`) and could not start.
Repointed at the local Python 3.11 install; `.venv/pyvenv.cfg.bak` holds the original.
`.venv/` is gitignored, so this is a local dev-env repair, not a repo change.

## 2. Existing M0–M5 components relevant to M6

| Component | State | How M6 uses it |
|---|---|---|
| `Pipeline.ask(question, scope)` | **Real and working**, all models stubbed | `eval/run_eval.py` calls it **unmodified** (its own docstring names M6 as a caller) |
| `Pipeline.ingest(path, ...)` | Real | `run_eval.py` ingests gold-set corpus docs so a `Scope` can exist |
| `AnswerResult{answer, decision, trace}` | Real | The unit `run_eval.py` scores |
| `Scope` (I1) | Real, enforced | Every eval retrieval call passes an explicit Scope — no exceptions for eval |
| `PipelineTrace.to_dict()` | Real | Cost/stage timings embedded in results files alongside metrics |
| `Claim{text, passage_ids, citations}` | Real | Gold groundedness/entailment schemas mirror these field names — no translation layer |
| `VerificationResult{status, score, reason}` | Real, I2-shaped | `status == "unavailable"` propagates into results as `null`, never a number |
| `StubEncoder/Reranker/Generator/Verifier` | Real | The "untrained system" the done-when criterion measures |
| `BackendInterface.fetch_chunks_scoped` | Real (local backend) | Retrieval-pooling source once M2 lands; today the stub path |
| M4 `registry/`, M5 `verify/`, `gate.py` | **Do not exist** | M6 must not create stubs of them (onboarding section 13) |
| `data/splits/*.json` | **Do not exist** | Leakage checks must skip honestly with a stated reason |

## 3. Existing evaluation functionality already present

`src/smartlawai/eval/metrics.py` — one module, 10 public functions, imported by exactly
two test files and nothing in `src/` or `api/` (re-grepped; the onboarding's section 4
snapshot is still accurate):

| Function | Measures | Verdict |
|---|---|---|
| `rouge_l(pred, ref)` | Hand-rolled LCS F1 | **DELETE** — `PLAN.md` §M6 is explicit |
| `token_f1(pred, ref)` | Bag-of-tokens F1 | Keep verbatim |
| `prf(tp, fp, fn)` | P/R/F1 from counts | Keep verbatim |
| `bertscore_f1(preds, refs)` | BERTScore F1 via `bert_score` | Keep verbatim (lazy import) |
| `ner_f1`, `ner_per_label_f1` | Entity P/R/F1 | Keep verbatim |
| `mean_reciprocal_rank` | MRR | Keep verbatim |
| `recall_at_k`, `precision_at_k` | Retrieval | Keep verbatim |
| `citation_recall`, `citation_precision` | Citation coverage | Keep verbatim |

Also present but **not** general evaluation code, and **out of M6's scope**:
`src/smartlawai/core/faithfulness.py` (an M0-era heuristic), `data/score_filter_pairs.py`
(M1's HHEM filter for Path-2 pairs), `data/pilot/hhem_score*.py`. None are touched.

`adapters/base.py` defines an `EvalResult` DTO and `BackendInterface.store_eval()` /
`fetch_annotations()`. These are M0's DuckDB-facing hooks. M6 writes JSON/JSONL files
to `eval/results/` and `eval/gold/` instead, because `OPS.md`'s retention table is
explicit that `eval/results/*.json` is kept **forever, in git** — a DuckDB row in a
gitignored `.smartlaw_local/` satisfies neither. The DTO is left untouched for whoever
wires the DB path later.

## 4. Existing files / classes / functions M6 reuses (does not rebuild)

- `Pipeline` — the whole retrieve/rerank/generate/verify/decide sequence. Not reimplemented.
- `Scope`, `PipelineTrace`, `Claim`, `GenerationResult`, `VerificationResult`.
- All 9 keeper metric functions above, moved not rewritten.
- `sample_docs/sample_judgment.txt` — structural smoke input only, never a reported number.
- `data/raw/in_ext_source/dataset/IN-Abs/{train,test}-data/{judgement,summary}/*.txt`
  (7,030 train docs, verified present) — perturbation source corpus. Plain text, so the
  generator needs no `datasets` dependency and no re-pull.
- `pyproject.toml`'s existing `[ui] streamlit>=1.30` — the annotation app's dependency;
  no second pin added.
- M1's `pytest.skip("... not landed")` convention for checks that depend on frozen splits.

## 5. Existing pipeline input/output formats

```python
Pipeline.ingest(path: str, doc_type: str, source: str, owner_id: str)
    -> tuple[doc_id: str, PipelineTrace]

Pipeline.ask(question: str, scope: Scope) -> AnswerResult
AnswerResult = {answer: str, decision: "ANSWER" | "REFUSE", trace: PipelineTrace}
```

`AnswerResult` carries no structured claim list — the claims live on the
`GenerationResult` consumed inside `ask()`, and reach the outside world only through
`trace.generation` (`{model, n_claims, unanswerable_aspects}`) and
`trace.verification.results`. **Consequence for M6:** claim-level metrics that need the
claim *texts* cannot be computed from `AnswerResult` alone today. M6 handles this
without touching `pipeline.py`: `run_eval.py` scores what the trace exposes (decision,
counts, verification status, retrieval ids, cost) and computes claim-level groundedness
from the **gold** `(passage, claim, label)` sets, which is exactly what those sets are
for. This is recorded as a known limitation, not worked around by widening a public API.

## 6. Trace / result objects available to evaluation

`PipelineTrace.to_dict()` yields:
`trace_id`, `scope`, `stages[]{name, status, started_at, duration_ms, detail, error}`,
`retrieval{n_candidates, candidate_chunk_ids, top_chunk_ids}`,
`generation{model, n_claims, unanswerable_aspects}`,
`verification{results[]{status, score, reason}}`,
`decision{outcome, reason}`, `cost{...}`, `created_at`.

This is enough for: retrieval metrics (`top_chunk_ids` vs gold relevant chunk ids),
refusal metrics (`decision.outcome` vs gold OOS label), calibration
(`verification.results[].score` as confidence), and latency/cost reporting
(`stages[].duration_ms`, `cost`).

## 7. Existing metrics and what they measure

See section 3's table. Coverage gaps against `RESEARCH.md` §5's metric table:

| RESEARCH.md §5 requires | Exists today? |
|---|---|
| Recall@{5,10,20}, MRR | yes |
| **nDCG@10** | missing |
| ROUGE-1/2/L via `rouge-score` | no — hand-rolled ROUGE-L only |
| BERTScore | yes (dependency not installed — section 10) |
| **Extractiveness (4-gram)** | missing |
| Claim-level groundedness | missing |
| **Citation validity rate** | missing |
| **Repealed-citation rate** | missing |
| Judge agreement (Cohen's kappa) | missing |
| **Per-perturbation-type detection rate** | missing |
| Refusal P/R on labelled OOS | missing (`prf` exists as a primitive) |
| **ECE + reliability diagram** | missing |
| Cost/latency p50/p95 | missing (trace has the raw numbers) |
| Paired bootstrap, Bonferroni | missing |

## 8. M6 requirements extracted from `M6_ONBOARDING.md`

1. **§4 — relocate** `src/smartlawai/eval/metrics.py` to `eval/metrics.py`; update the two
   test importers; delete the old package. Note it in the commit message.
2. **§8 — `eval/metrics.py`**: delete `rouge_l`, add `rouge_scores()` backed by
   `rouge-score`; keep the other nine green; add `citation_validity_rate` and
   `repealed_citation_rate` taking *already-resolved* registry-result dicts — **no import
   of M4's `verify/registry_check.py`**, which does not exist.
3. **§8 — `eval/stats.py`**: `paired_bootstrap` (10k resamples, seed 42, 95% CI),
   `expected_calibration_error`, `reliability_diagram_data`, `bonferroni_correct`.
4. **§8 — `eval/extractiveness.py`**: fraction of pred's 4-grams appearing verbatim in source.
5. **§6 — annotation app** (`eval/annotate/app.py`), built **first**: Streamlit, one item
   per screen, randomised order, blinded system labels, correction vs cold mode, logs
   rater id / timestamp / **time-per-item**, live kappa/alpha, exports straight to
   `eval/gold/*.jsonl`, annotator-pool-agnostic.
6. **§6 — `eval/annotate/agreement.py`**: Cohen's kappa and Krippendorff's alpha, used
   live by the app and standalone when reporting a study (`RESEARCH.md` §7.5).
7. **§7 — five gold sets** with per-set IAA duplication rates (100/30/100/0/100 %) and a
   stated **power calculation**, not a round number. `retrieval.jsonl` built by **TREC
   pooling** over BM25/dense/hybrid top-10 (~15 unique candidates/query).
8. **§9 — `data/perturbations/generate_perturbations.py`**: seven legal perturbation types
   over IN-Abs, seeded, releasable standalone, every row tagged with its source `doc_id`,
   provably disjoint from every eval split (train-split-only once splits land).
9. **§10 — ClaimRAG-Law**: locate and cache read-only, or state explicitly that it is not
   publicly available. **No silent substitution.**
10. **§11 — `eval/run_eval.py`**: run `Pipeline` over a split, compute every applicable
    metric, write `eval/results/{split}_{timestamp}.json`. Never hand-type a number.
11. **§11 — CI eval-regression gate** on a fixed dev subset, generous initial threshold.
12. **§14 — `pyproject.toml`** gains an `[eval]` extras group; `.gitignore` gains the
    generated `perturbations.jsonl` **only** (gold sets and results stay committed).
13. **Done when:** `run_eval.py` produces a complete results table end to end on the
    untrained stub system. Ugly numbers are fine; a broken harness is not.

## 9. Requirements inferred from the two reference images

**Image 1 (project plan).** M6 sits in two phases: a *preparation* phase that runs in
parallel with M0–M5 (test sets, annotation guidelines, ground truth, rating interface,
scoring scripts) and a *final execution* phase that needs a working M2–M5 pipeline. The
harness must therefore run **before** the pipeline is real — which matches the done-when
criterion and is why `run_eval.py` must degrade gracefully rather than assume real models.

**Image 2 (M6 architecture & workflow).** Adds four things the prose implies but does not
lay out as a diagram, all adopted:

- The four-block decomposition of the evaluation system — **Evaluation Runner /
  Automatic Metrics / Human Evaluation / Statistical Analysis** — becomes the module
  boundary: `run_eval.py` / `metrics.py` + `extractiveness.py` / `annotate/` / `stats.py`.
- **Deliverables** list names *evaluation report (tables + results)* as a first-class
  output, so `run_eval.py` emits machine-readable JSON **and** a human-readable Markdown
  report, not just JSON.
- **Leakage check scripts — "graceful skip if splits not available"** — confirms M1's
  skip convention is the intended behaviour, not a workaround.
- The annotation-app panel names *guidelines & rating scales* as visible in the UI, so
  rubrics are data (`eval/annotate/guidelines.py`), rendered in-app, versioned with the
  export, not prose in a README the rater never opens.
- **Phase workflow**: docs -> design -> **app -> gold data -> metrics -> end-to-end run ->
  report**. The app precedes the metrics. Honoured.
- Key-principles panel repeats: never fall back to fabricated scores (I2), use frozen
  splits, use standard metric libraries, ensure perturbations never overlap eval splits,
  keep logging/reproducibility/versioning. All already invariants here.

Nothing in either image contradicts `PLAN.md` / `RESEARCH.md`.

## 10. Missing components required to complete M6

| Missing | Where it will live |
|---|---|
| Top-level `eval/` package + path resolution for tests | `eval/__init__.py` (repo root is already on `sys.path` under pytest via rootdir insertion) |
| Gold-set schemas + validation + loader | `eval/schemas.py`, `eval/gold_sets.py` |
| `rouge_scores`, `ndcg_at_k`, groundedness/refusal/citation-validity/repealed/perturbation metrics | `eval/metrics.py` |
| Paired bootstrap, ECE, reliability data, Bonferroni, power calculation | `eval/stats.py` |
| Extractiveness | `eval/extractiveness.py` |
| Cohen's kappa, Krippendorff's alpha, anchoring check | `eval/annotate/agreement.py` |
| Annotation storage (append-only JSONL + export) | `eval/annotate/store.py` |
| Rubrics / rating scales as data | `eval/annotate/guidelines.py` |
| Streamlit annotation app | `eval/annotate/app.py` |
| TREC pooling for `retrieval.jsonl` candidates | `eval/annotate/pooling.py` |
| Runner + reporting | `eval/run_eval.py`, `eval/report.py` |
| Perturbation generator | `data/perturbations/generate_perturbations.py` |
| Leakage / disjointness checks | `tests/test_perturbations.py` |
| `[eval]` extras, `.gitignore` line, CI gate | `pyproject.toml`, `.gitignore`, `.github/workflows/eval-regression.yml` (new file) |
| ClaimRAG-Law provenance decision | `eval/gold/external/PROVENANCE.md` |

**Dependency reality check (onboarding §18 asked for this explicitly):** `scipy` **is**
installed; `rouge-score` and `streamlit` were **not** and have now been installed;
**`bert_score` is not installed and `torch` is not present**, and no existing test
exercises `bertscore_f1`. So the answer to §18's open question is *no, BERTScore has
never run in this repo*. `bertscore_f1` is kept unchanged with its lazy import, and its
new test skips with that reason stated rather than pretending to pass.

## 11. Proposed M6 architecture

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
      source: data/raw/.../IN-Abs   guard: tests/test_perturbations.py (disjointness)
```

Layering rule, enforced by a test: `metrics` / `stats` / `extractiveness` / `agreement` /
`schemas` import **nothing** from `smartlawai` — they are pure functions over data
structures, exactly as `OPS.md` §8 promises for Track D. Only `run_eval.py` (and the app,
for smoke input) may import `Pipeline`.

**I2 in the harness.** Every metric returns `None` rather than a number when its input is
unavailable, and the results JSON records a per-metric `status` of `ok` /
`unavailable` / `not_applicable` with a reason. A missing verifier score never becomes
`0.5`, and an empty gold set never becomes a `0.0` that reads like a measured result.

## 12. Exact integration points with the existing pipeline

| # | Integration point | Direction | Contract |
|---|---|---|---|
| 1 | `run_eval.py` -> `Pipeline.ask(question, Scope(...))` | read | Public method, unmodified. Scope built per gold item from its `doc_ids` + a fixed eval `owner_id`. |
| 2 | `run_eval.py` -> `Pipeline.ingest(...)` | write | Only into a throwaway `SMARTLAW_LOCAL_DIR`, so eval never pollutes a real store. |
| 3 | `AnswerResult.trace.retrieval["top_chunk_ids"]` -> retrieval metrics | read | Existing trace field. |
| 4 | `AnswerResult.decision` -> refusal / OOS metrics | read | `"ANSWER"` / `"REFUSE"`. |
| 5 | `trace.verification["results"][].score/status` -> ECE + I2 propagation | read | `status="unavailable"` means the item is excluded from calibration and counted, never defaulted. |
| 6 | `trace.stages[].duration_ms`, `trace.cost` -> latency/cost table | read | Existing. |
| 7 | `Claim{text, passage_ids, citations}` -> gold schema field names | naming | Gold groundedness/entailment rows use `passage_id` / `claim_text` / `citations` so no translation layer is needed when M3/M5 expose real claims. |
| 8 | `citation_validity_rate` / `repealed_citation_rate` <- M4 registry results | future | Accepts already-resolved dicts (`{citation, exists, in_force, superseded_by}`). M6 creates **no** import edge to `registry/`. |
| 9 | `tests/test_data_leakage.py` (M1, not yet written) globs `eval/gold/*.jsonl` | future | M6's gold rows carry `source_doc_id` so M1's planned helper works unchanged the moment it lands. |

**Zero changes to `pipeline.py`, `scope.py`, `trace.py`, `protocols.py`, `adapters/*`,
`api/main.py`, `ui/app.py`.**

## 13. Files that should be created

```
eval/__init__.py, schemas.py, gold_sets.py, metrics.py (moved+edited), stats.py,
     extractiveness.py, report.py, run_eval.py, README.md
eval/annotate/__init__.py, agreement.py, guidelines.py, pooling.py, store.py, app.py
eval/gold/{retrieval,oos,groundedness,repealed,entailment}.jsonl  + PROVENANCE.md
eval/gold/external/PROVENANCE.md
eval/results/.gitkeep
data/perturbations/generate_perturbations.py, README.md
tests/test_eval_schemas.py, test_gold_sets.py, test_stats.py, test_extractiveness.py,
      test_agreement.py, test_annotation_store.py, test_pooling.py,
      test_run_eval.py, test_report.py, test_perturbations.py, test_eval_integration.py
docs/M6_IMPLEMENTATION_ANALYSIS.md, M6_IMPLEMENTATION_PLAN.md, M6_COMPLETION_REPORT.md
.github/workflows/eval-regression.yml   (new file; no existing workflow edited)
```

## 14. Existing files that may need modification

| File | Change | Why |
|---|---|---|
| `tests/test_metrics.py` | import path -> `eval.metrics`; ROUGE tests target `rouge_scores` | Onboarding §4 / §8 |
| `tests/test_local_backend.py` (1 line) | import path -> `eval.metrics`, `rouge_l` -> `rouge_scores` | Same |
| `src/smartlawai/eval/` | **deleted** after the move | Onboarding §4.3 — don't leave a duplicate |
| `pyproject.toml` | add `[eval]` extras group | Onboarding §14 |
| `.gitignore` | ignore `data/perturbations/perturbations.jsonl` only | Onboarding §14 |
| `conftest.py` | ensure the repo root is importable so `import eval.metrics` resolves | Consequence of the §4 move |

## 15. Files / modules that must NOT be modified

`src/smartlawai/pipeline.py`, `scope.py`, `trace.py`, `protocols.py`, `config.py`,
`adapters/*`, `core/*`, `guardrails/*`, `api/main.py`, `ui/app.py`, `cli.py`,
`train/**`, `gcloud/**`, `docker/**`, the five existing `.github/workflows/*.yml`,
everything under `data/` except the new `data/perturbations/`, and
`registry/` / `verify/` (M4/M5 territory — not even as stubs).

`tests/test_scope.py::test_cross_document_isolation` is never deleted or skipped
(CLAUDE.md §4).

## 16. Conflicts found, and how they were resolved

1. **`OPS.md` §8 says Track D "never imports the pipeline"; `M6_ONBOARDING.md` §11–12 says
   `run_eval.py` calls `Pipeline.ask()` directly.** Resolved by layering: the metric layer
   imports nothing from `smartlawai` (a test enforces it), and the single import edge lives
   in `run_eval.py`, which is the orchestrator the onboarding explicitly describes.
2. **`AnswerResult` exposes no claim texts** (section 5). Resolved by scoring the trace for
   system-level metrics and using gold `(passage, claim)` sets for claim-level ones; no
   public API widened. Recorded as a limitation for M3/M5 to close.
3. **`bert_score` is not installed** (section 10). Resolved by keeping the function, testing
   it under `importorskip`, and stating the fact rather than silently dropping the metric.
4. **Gold sets require human annotators M6 cannot summon.** Resolved by shipping the full
   collection machinery plus small, honestly-labelled **seed** sets whose provenance notes
   say exactly what they are and what a real study must replace them with. No fabricated
   human ratings, no invented kappa.
