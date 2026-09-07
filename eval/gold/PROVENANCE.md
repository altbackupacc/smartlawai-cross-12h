# `eval/gold/` — provenance

Same discipline as `data/splits/*.json` (I5-adjacent, `M6_ONBOARDING.md` §14):
these files are **frozen reference data**. Commit them; do not regenerate them
silently. Every load records a sha256, and every `eval/results/*.json` embeds
those checksums, so an undocumented edit is detectable by comparison.

---

## Read this first: what is shipped here is a SEED, not the study

`RESEARCH.md` §7 specifies 300–400 items per set, produced by multiple raters
through `eval/annotate/app.py`, each set reporting an inter-annotator agreement
statistic. **That study has not been run.** M6 delivers the machinery for it
(the app, the rubrics, the storage, the pooling, the agreement statistics, the
export) plus small seed sets so the harness runs end to end on real data today.

Every seed row carries `n_annotations: 1`, `agreement: null`, and a `provenance`
string stating how its label was arrived at. **No human study is claimed and no
agreement number is invented.** A run's report prints each set's achieved *n*
next to its target and the resulting CI half-width, so a shortfall is visible in
the numbers rather than buried in this file.

Regenerate the seeds with `python -m eval.build_seed_gold`. Replace them by
running the app and exporting — the export overwrites these files.

## Current state

| Set | Seed n | Target n (`RESEARCH.md` §7.1.4) | 95% CI half-width at p=0.5 | IAA duplication achieved / required |
|---|---:|---:|---:|---:|
| `retrieval.jsonl` | 10 queries | 200 | ±31.0% | 0% / 100% |
| `oos.jsonl` | 32 | 350 | ±17.3% | 0% / 30% |
| `groundedness.jsonl` | 40 | 350 | ±15.5% | 0% / 100% |
| `repealed.jsonl` | 34 | 350 | ±16.8% | 0% / 0% (objective by design) |
| `entailment.jsonl` | 40 | 350 | ±15.5% | 0% / 100% |

**The power calculation, stated rather than assumed** (`RESEARCH.md` §7.1.4):
for a proportion, the 95% CI half-width is `1.96·√(p(1−p)/n)`, maximised at
p=0.5. That gives ±5.7pp at n=300 and ±4.4pp at n=500 — the "n≈300 → ±5%,
n≈500 → ±4%, diminishing returns beyond" the protocol describes. Targets here
are 350 per set (200 *queries* for retrieval, each carrying ~15 pooled
judgments). `eval/gold_sets.py::proportion_ci_halfwidth` computes this from each
set's **achieved** n, so a short set reports its real, wider interval.

---

## `retrieval.jsonl` — queries only, judgments deliberately empty

Ten queries against the sample judgment corpus, with `judgments: {}`.

The empty pool is a finding, not an omission. `doc_id` and `chunk_id` are fresh
UUIDs on every ingest (`src/smartlawai/core/ocr.py:53`,
`src/smartlawai/core/chunking.py:52`), so chunk ids committed today refer to
nothing tomorrow. Relevance judgments only become meaningful once M2 lands a
stable corpus with stable ids.

Consequence, visible in every report: retrieval metrics report
`unavailable: no_relevance_judgments` rather than a fabricated `0.0`.

**How to populate it, once ids are stable** (`RESEARCH.md` §7.1.2, TREC pooling):
run BM25, dense and hybrid retrieval for each query, pool each method's top-10
via `eval/annotate/pooling.py::pool_candidates`, write the ~15 unique candidates
per query to a queue file, and judge them in the app. That is ~3,000 judgments
for 200 queries instead of millions.

## `oos.jsonl` — three-way scope classification

32 curated questions: 12 `in_scope`, 10 `out_of_scope`, 10 `advice_seeking`.
Labelled by the M6 implementer against `eval/annotate/guidelines.py`'s OOS
rubric (`v1.0.0`). The distinction is largely objective, which is why
`RESEARCH.md` §7.1.3 duplicates only 30% of this set — but 30% of 32 rows have
not been duplicated, so **no agreement statistic exists for it yet**.

Feeds M5's OOS threshold calibration (`PLAN.md` M5.6). M6 produces the labelled
set; it calibrates nothing.

## `groundedness.jsonl` and `entailment.jsonl` — separate files, separate purposes

Identical schema, different jobs, and **they must never be merged**:

- `groundedness.jsonl` is the **evaluation** set that scores the running system's
  own claim/passage pairs.
- `entailment.jsonl` is the **held-out judge-evaluation** set that measures how
  good InLegalNLI and HHEM are as classifiers. `RESEARCH.md` §2.5's hard rules:
  **never used to train InLegalNLI.** `EntailmentItem` is a distinct class
  carrying `never_train_on=True` in `GOLD_SET_SPECS`, and
  `tests/test_perturbations.py` asserts no perturbation training row derives
  from a document appearing here.

Both are 40 rows (20 positive / 20 negative) built from real **IN-Abs
test-data** judgments:

- **Positives** are headnote sentences whose content words are ≥85% present in
  the matched judgment passage — near-verbatim, so entailment is checkable by
  reading rather than asserted.
- **Negatives** are those same pairs with exactly one `RESEARCH.md` §2.5
  perturbation applied (recorded per row in `notes`), which makes them
  not-entailed **by construction**: the claim now asserts a section number,
  party, date, court or holding the passage does not contain.

Source documents come from IN-Abs **test-data** while `data/perturbations/`
reads **train-data**. The two folders share no document ids (verified: 7,030 vs
100 stems, zero overlap), so `RESEARCH.md` §2.5's disjointness rule holds by
construction and its test is an enforced assertion today rather than a skip.

**What a real study still has to do here:** 100% duplication for IAA, a 20-item
pilot with a rubric revision, an anchoring check on a 10% cold subset, and
expansion to ~350 rows drawn from more than the handful of documents these 40
came from.

## `repealed.jsonl` — the IPC → BNS gold set

34 criminal- and civil-law queries: 26 labelled `repealed`, 8 `in_force`. Covers
IPC→BNS, CrPC→BNSS and Indian Evidence Act→BSA (all in force from 1 July 2024),
plus in-force controls from the Constitution, Contract Act and Limitation Act so
the set is not all-positive.

0% duplicated for IAA **by design** — it is an objective registry lookup and the
objectivity is the point (`RESEARCH.md` §7.1.3). A report renders this set's
agreement as `not_applicable` with that reason, not as `unavailable`.

> **⚠ Unverified successor mappings.** "Objective" means "checkable against a
> primary source", and these seed rows have **not** been checked row-by-row
> against the Gazette of India. Successor sections were taken from the replacing
> Acts' own section text. Every row says so in its `authority_source` field.
> Verifying them against primary sources is the first task of whoever runs this
> set for real; M4's registry is what will do it at scale. **Do not publish a
> number computed from this set until that verification is done.**

Release-ready shape from the start (`PLAN.md` M6): every row carries its own
`as_of_date` and `authority_source`, so publishing this as a standalone
benchmark later is a licensing decision, not rework.

## `external/` — third-party sets

See `external/PROVENANCE.md`. Nothing in that directory is relabelled or
modified; its value is being independent of anything built in this project.
