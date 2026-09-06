# M6 Onboarding — Evaluation Harness + Gold Sets

You're picking up **M6** on `smartlawai` in a fresh Claude Code chat with no access to
any prior conversation. This doc is everything you need to start without re-deriving
context from anyone. Read it top to bottom once, then use it as a reference. It is
scoped to M6 exclusively — it does not cover M0 (merged), M1 (in progress, owned
elsewhere), M4/M5 (not started), or GCP/training infrastructure.

**No GPU needed for any of this.** M6 is pure Python, mostly CPU-bound metric code
plus a human-facing Streamlit annotation app. Don't touch Cloud Shell, `gcloud`, or
Docker. The only real cost here is human annotator time, not compute.

**M6 is fully standalone** (`OPS.md` §8 names it explicitly, alongside M4, as one of
the two tracks that "need nothing from anyone"). You do not need M1's data pipeline
finished, M4's registry built, or M5's gate wired up to do the great majority of this
work. Where a genuine dependency exists, it's called out explicitly below (§3) with a
way to make progress anyway.

---

## 1. Read these four files in the repo root first, in this order

1. **`CLAUDE.md`** — non-negotiable invariants, repo layout, conventions. Short, read
   it fully. **I2** (verifiers never fail open into a fabricated score) and **I5**
   (frozen splits) both bear directly on what you're building.
2. **`PLAN.md`** — find the `## M6` section. That's your literal spec: what to build,
   the annotation techniques to use, and the exact "done when" criterion. This doc
   restates and expands it below; `PLAN.md` is the source of truth if anything here
   seems to drift from it.
3. **`RESEARCH.md`** — read **§5 (METRICS)**, **§6 (STATISTICAL PROTOCOL)**, and
   **§7 (ANNOTATION PROTOCOL)** in full before writing a line of code. Skim **§8**'s
   threats **T3** (metric gaming), **T4** (silver data), **T9** (annotator expertise).
   These three sections *are* M6 — the milestone exists to operationalize them.
4. **`OPS.md`** §8 ("Parallel Work") — explains why M6 was picked as a standalone
   track and what it hands off to (and receives from) the other tracks.

---

## 2. What you're actually building (summary of `PLAN.md`'s M6)

The scoring system for the entire project, and — the harder, more important half —
the human-labelled test sets that scoring depends on. Nothing else in this project can
claim to work without this milestone; M7's baselines and M8's training are both
explicitly gated on it (`OPS.md` §8: *"Nobody submits a training job until Track D
[M6] can measure it."*).

Concretely, four things:

1. **`eval/metrics.py`** and **`eval/stats.py`** — standard metric implementations
   (never hand-rolled where a real library exists) plus the statistical machinery
   (paired bootstrap, calibration) that turns raw metric numbers into defensible
   claims.
2. **The annotation app** (~4h to build, per `PLAN.md` — build this *first*, everything
   else depends on it) — a Streamlit tool that makes human labelling fast, consistent,
   and methodologically sound (blinded, randomised, logged, live-agreement-tracked).
3. **Five gold test sets** in `eval/gold/*.jsonl`, produced using that app:
   `retrieval.jsonl`, `oos.jsonl`, `groundedness.jsonl`, `repealed.jsonl`,
   `entailment.jsonl`.
4. **`data/perturbations/`** — a synthetic-corruption generator over IN-Abs claims,
   used later (M8a) to train InLegalNLI. Not evaluation data — training data for a
   *different* milestone's model — but M6 builds and owns the generator, and is
   responsible for proving it never overlaps any evaluation split.

**Done when** (verbatim from `PLAN.md`): *"the full table computes for the untrained
system. That is your floor."* — i.e., `eval/run_eval.py` produces a complete results
table by running the (currently all-stub) M0 pipeline through every metric, end to
end, with real gold data. It doesn't need to look good. It needs to run and be honest.

---

## 3. What you're inheriting — the current state of the rest of the project

Read this before assuming any other milestone's output exists.

- **M1 (data foundation) is mid-flight, not finished.** As of this writing:
  `data/raw/{summ,rr,lner,lsi}` are pulled and cached. The RR↔SUMM alignment check
  ran and found **zero document overlap between the two configs** — this forced
  `PLAN.md`'s fallback path (**Path 2**: frontier-model-generated section summaries,
  human-validated). A dry-run cost estimate exists
  (`data/processed/path2_cost_estimate.json`, ~35,650 expected section pairs, ~$25–150
  depending on model tier) but the actual generation run, dedup, and
  `data/splits/{train,dev,test}.json` **have not been produced yet**. There is no
  `tests/test_data_leakage.py` yet either.
- **What this means for you**: you cannot assume `data/splits/*.json` exists. Build
  every piece of code that would consume it (leakage checks, "is this eval item from a
  training doc" checks) to **degrade gracefully with an explicit skip-and-state-why**,
  exactly like M1's own leakage-test design already does elsewhere in this project —
  don't invent a different convention. See §9.
- **What you can do regardless**: build the annotation app, build the five gold sets'
  *schemas and app-driven collection workflow*, build `eval/metrics.py`/`stats.py`,
  build the perturbation generator against the already-pulled `data/raw/summ`. None of
  that needs frozen splits to exist as code — only the final "prove no leakage" check
  needs them, and that check is designed to wait honestly rather than block you.
- **M4 (authority registry) has not started.** `registry/` and `verify/` don't exist.
  `repealed.jsonl` (§6 below) is a set of criminal-law queries with human judgments —
  you can build and populate it now; it doesn't need the registry to exist, only to
  eventually be *checked against* it (that's M9's job, using your gold set).
- **M5 (verification + gate) has not started.** `gate.py` doesn't exist.
  `groundedness.jsonl` and `entailment.jsonl` likewise don't need it to exist yet —
  they're human judgments about (passage, claim) pairs, independent of what verifier
  eventually scores them.
- **The M0 pipeline is real and running.** `Pipeline` (in `src/smartlawai/pipeline.py`)
  works end to end today with every model stubbed (`StubEncoder`/`StubGenerator`/
  `StubVerifier`). `eval/run_eval.py` should call this class directly, unmodified —
  its own docstring says as much: *"a later research/eval harness (M6) calls the same
  class, unmodified."* This is why the done-when criterion only asks for the table to
  compute on the untrained system — that's what exists right now.

---

## 4. A location decision you need to make on day one

`PLAN.md`'s repo layout diagram (`CLAUDE.md` §3) shows evaluation code and data
living together at the **top level**: `eval/metrics.py`, `eval/gold/`,
`eval/run_eval.py`. But a metrics module already exists today at
**`src/smartlawai/eval/metrics.py`** — a leftover from the original v1 import,
flagged explicitly in `M1_ONBOARDING.md` §9 as *"a pre-existing inconsistency... out
of scope for M1... belongs to whoever eventually does the M6 eval-harness work."*
That's you.

Two files import it today, both tests:
```
tests/test_metrics.py:4    from smartlawai.eval.metrics import (...)
tests/test_local_backend.py:50   from smartlawai.eval.metrics import prf, rouge_l, token_f1
```
Nothing in `src/` or `api/` imports it — it's not wired into the running pipeline
anywhere yet.

**Recommended resolution** (confirm this is genuinely the only two importers before
you rely on it — re-grep `smartlawai.eval` across the repo, don't just trust this
document):
1. Move the file to the top-level location `PLAN.md` specifies: `eval/metrics.py`.
2. Update the two test imports above to `from eval.metrics import ...` (or however
   your `conftest.py`/`sys.path` setup resolves top-level `eval/` — check how the
   existing top-level `eval/` files from M1, if `eval/ocr_wer.py` has landed by the
   time you start, resolve their own imports, and match that pattern for
   consistency).
3. Delete `src/smartlawai/eval/` (both `__init__.py` and the old `metrics.py`) once
   the move is confirmed working — don't leave a duplicate.
4. Note the move plainly in your first commit message so it's easy to trace in
   history.

This is a judgment call, not a hard requirement — if you find a reason mid-work that
top-level `eval/` genuinely doesn't fit (e.g. packaging concerns), say so explicitly
rather than silently leaving the inconsistency for a third person to inherit.

---

## 5. Directory and file layout (target state)

```
eval/
├── metrics.py                 # moved per §4; rouge-score + bert-score, hand-rolled
│                               #   ROUGE-L deleted per PLAN.md's explicit instruction
├── stats.py                   # paired bootstrap, ECE, reliability diagrams (§8 below)
├── extractiveness.py          # 4-gram overlap with source (§8 below)
├── run_eval.py                # orchestrator: Pipeline over a split -> results table
├── annotate/
│   ├── app.py                 # the Streamlit annotation tool (§6 below) -- build first
│   └── agreement.py           # Cohen's kappa / Krippendorff's alpha, used live by app.py
│                               #   and again standalone when reporting each study (§7.5 RESEARCH.md)
├── gold/
│   ├── retrieval.jsonl        # ~200 queries + pooled relevant chunk ids
│   ├── oos.jsonl              # labelled in-scope / OOS / advice-seeking
│   ├── groundedness.jsonl     # claim-level (passage, claim, supported?)
│   ├── repealed.jsonl         # criminal-law queries for the IPC->BNS experiment
│   └── entailment.jsonl       # human-labelled (passage, claim, entailed?) -- NEVER trained on
├── results/                   # eval/run_eval.py output -- committed, kept forever (OPS.md §4 retention table)
└── (eval/ocr_wer.py, eval/ocr_wer.md live here too, if M1 has landed them by the time you start -- don't touch)

data/
└── perturbations/
    ├── generate_perturbations.py   # the releasable generator (§9 below)
    └── perturbations.jsonl          # generated output -- gitignored if large; regenerable from the script + seed

tests/
├── test_metrics.py            # existing (moved per §4), extended with rouge-score/bert-score tests
├── test_stats.py              # new: bootstrap CI, ECE sanity checks
├── test_extractiveness.py     # new
└── test_perturbations.py      # new: generator produces valid, disjoint-from-splits output
```

---

## 6. The annotation app — build this first (`eval/annotate/app.py`)

Per `PLAN.md`, this is a ~4-hour build and *everything else depends on it* — the five
gold sets don't get produced without it. Streamlit, keyboard-driven, one item per
screen.

**Must enforce** (from `PLAN.md` and `RESEARCH.md` §7.2 — these aren't nice-to-haves,
they're what makes the resulting data usable in a paper):

- **Randomised item order + blinded system labels.** A rater who can tell which
  output is "ours" is not giving you clean data. Never surface a model/system
  identifier in the UI.
- **Two modes**: *correction mode* (a model-proposed label is pre-filled; rater
  accepts or edits) and *cold mode* (blank, rater labels from scratch). You need both
  — correction mode gives 2–3× throughput (`RESEARCH.md` §7.1), but a 10% cold subset
  is required to detect anchoring (do assisted labels agree with the model far more
  than the cold subset does? If so, report and discount).
- **Logs rater id, timestamp, and time-per-item** for every submission — time-per-item
  is what sizes every later gold set (§7.3 below).
- **Live Cohen's κ / Krippendorff's α** — computed via `eval/annotate/agreement.py` —
  shown to whoever's running the session as items come in, not just at the end.
- **Exports straight to `eval/gold/*.jsonl`** — no manual spreadsheet-to-JSON step.

**Keep it annotator-pool-agnostic.** Nothing in the app or the export schema should
assume who's rating (students vs. practising advocates — `RESEARCH.md` §7.4 uses
both). This is what makes recruiting advocates later a config change, not a rebuild
(see `PLAN.md`'s `## V2` section).

---

## 7. The five gold sets — what each one is and how it's produced

All five live in `eval/gold/*.jsonl`. Build the app (§6) once, then run each set
through it. Sizing and duplication rules below come straight from `RESEARCH.md` §7.1.

| Set | What it captures | Duplicate for IAA? | Approx N |
|---|---|---|---|
| `retrieval.jsonl` | Is this chunk relevant to this query? | ✅ 100% | ~200 queries |
| `oos.jsonl` | In-scope / out-of-scope / advice-seeking classification | ⚠️ 30% | 300–400 |
| `groundedness.jsonl` | Is this claim supported by its cited passage? | ✅ 100% | 300–400 |
| `repealed.jsonl` | Criminal-law queries — is the cited law still in force? | ❌ 0% (objective registry lookup) | 300–400 |
| `entailment.jsonl` | (passage, claim, entailed?) — held out, **never used to train InLegalNLI** | ✅ 100% | 300–400 |

**Sizing**: use a real power calculation, not a round number (`RESEARCH.md` §7.1.4) —
n≈300 gives ≈±5% CI on a proportion, n≈500 gives ≈±4%, diminishing returns beyond
that. State the calculation in whatever provenance note you write for each set, the
same way M1's `PROVENANCE.md` states its own methodology.

### `retrieval.jsonl` — use pooling, not exhaustive judgment

Judging every chunk against every query is impossible. Follow TREC pooling
(`RESEARCH.md` §7.1.2): run BM25, dense, and hybrid retrieval (even the M0 stub
versions, for now — the *queries* and *pooling logic* are what you're building; swap
in real retrieval once M2 lands), pool each method's top-10, and judge only the ~15
unique candidates per query. Reduces 200 queries from millions of judgments to ~3,000.

### `oos.jsonl` — three-way label

Each item: a question, labelled one of `in_scope` / `out_of_scope` /
`advice_seeking`. This feeds M5's OOS threshold calibration later (`PLAN.md` M5.6) —
you're not calibrating anything here, just producing the labelled set it calibrates
against.

### `groundedness.jsonl` and `entailment.jsonl` — nearly identical schema, different purpose

Both are `(passage, claim, label)` triples. The distinction matters:
- `groundedness.jsonl` is the *evaluation* set used to score the running system's
  actual claim/passage pairs.
- `entailment.jsonl` is a broader, independent **held-out judge-evaluation set** — it
  measures how good InLegalNLI/HHEM *are as classifiers*, and per `RESEARCH.md` §2.5's
  hard rules, **must never be used to train InLegalNLI**. Keep these as genuinely
  separate files with separate provenance, even though the schema looks the same —
  don't let someone later merge them by accident.

### `repealed.jsonl` — the IPC→BNS gold set

Criminal-law queries where the human-labelled correct answer states whether the
citation a system might produce (e.g. "IPC §420") is currently in force, and if not,
what superseded it. This is what `RESEARCH.md` §2's natural experiment measures
against in M9. Since it's "0% duplicate for IAA — it's an objective registry lookup,"
a single knowledgeable rater per item is fine; the objectivity is the whole point
(`RESEARCH.md` §7.1.3).

---

## 8. `eval/metrics.py` and `eval/stats.py`

### `eval/metrics.py`

What already exists (after the move in §4) and should be **kept**: `token_f1`, `prf`,
`bertscore_f1` (already calls the real `bert_score` package correctly — don't touch),
`ner_f1`, `ner_per_label_f1`, `mean_reciprocal_rank`, `recall_at_k`, `precision_at_k`,
`citation_recall`, `citation_precision`. All already have passing tests in
`tests/test_metrics.py`; keep them green through the move.

**What must change**: `rouge_l` is currently hand-rolled (a manual LCS dynamic-program
implementation). `PLAN.md` §M6 is explicit: *"Delete the hand-rolled ROUGE-L."*
Replace it with the real `rouge-score` package:
```python
from rouge_score import rouge_scorer

def rouge_scores(pred: str, ref: str) -> dict[str, float]:
    """ROUGE-1/2/L F1 via the rouge-score package (RESEARCH.md §5: 'never hand-rolled')."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}
```
Update `tests/test_metrics.py`'s ROUGE tests to call this instead of the deleted
`rouge_l`. Keep the exact-match / partial-match / empty-string test cases — they're
still valid assertions, just against the new function.

Add citation-validity and repealed-citation-rate metric functions matching
`RESEARCH.md` §5's metric table — `citation_validity_rate` and
`repealed_citation_rate` — even though they can't be *computed* against a real
registry until M4 lands. Write them to accept a list of already-resolved
`RegistryResult`-shaped dicts (don't import `verify/registry_check.py` — it doesn't
exist yet, and M6 must not create a hard dependency on M4's code landing first).
Document in a docstring that the caller (eventually M9's `run_eval.py` invocation)
is responsible for producing that resolved input once the registry exists.

### `eval/stats.py` — new file

Three things, all named explicitly in `RESEARCH.md` §5–6:

```python
def paired_bootstrap(scores_a: list[float], scores_b: list[float],
                      n_resamples: int = 10_000, seed: int = 42) -> dict:
    """Paired bootstrap over test items for comparing two systems on the same
    items (RESEARCH.md §6) -- NOT a t-test over 5 seeds, which is underpowered
    for this kind of comparison. Returns the mean difference and a 95% CI;
    a CI that excludes zero is the significance claim."""

def expected_calibration_error(confidences: list[float], correctness: list[bool],
                                n_bins: int = 10) -> float:
    """ECE: bin predictions by confidence, compare each bin's mean confidence
    to its actual accuracy. RESEARCH.md §5: mandatory because the system
    displays a faithfulness score to non-expert users making real decisions."""

def reliability_diagram_data(confidences: list[float], correctness: list[bool],
                              n_bins: int = 10) -> list[dict]:
    """Per-bin (mean_confidence, accuracy, count) -- what a reliability diagram
    plots. Returns data, not a rendered figure; plotting is run_eval.py's job."""
```

Apply **Bonferroni correction** when comparing more than two systems on one metric
(`RESEARCH.md` §6) — a helper like `bonferroni_correct(p_values: list[float],
alpha: float = 0.05) -> list[bool]` covers this; don't leave it as a manual step
someone has to remember in `run_eval.py`.

### `eval/extractiveness.py` — new file, small

```python
def extractiveness(pred: str, source: str, n: int = 4) -> float:
    """Fraction of pred's n-grams (default 4-gram) that appear verbatim in source.
    RESEARCH.md §5: mandatory alongside faithfulness -- HHEM rewards verbatim
    copying, so a model that just quotes the source scores near-perfect
    faithfulness while being a useless summariser. High extractiveness + high
    faithfulness together is the honest reviewer-facing answer to
    'did you just train a copier?'"""
```

---

## 9. `data/perturbations/` — the synthetic entailment training generator

This is training data for **M8a**'s InLegalNLI (not for anything M6 evaluates
against), but `PLAN.md` puts building it under M6, and it needs to be built with
evaluation-split-safety in mind from day one — which is why it's your job and not
M8a's.

Generate corrupted-claim pairs from IN-Abs (`data/raw/summ`, already pulled by M1 —
you can start on this without waiting for M1's dedup/split step to finish, since this
generator reads raw documents, not the frozen split files). Corrupt genuinely-entailed
`(passage, claim)` pairs along the seven legal-specific perturbation types in
`RESEARCH.md` §2.5:

| Perturbation | Legal failure mode it simulates |
|---|---|
| Change section number | Wrong statutory provision |
| Substitute IPC↔BNS incorrectly | Repealed / superseded law |
| Negate obligation (`shall` → `shall not`) | Reversed legal effect |
| Swap appellant / respondent | Misattributed holding |
| Alter date or monetary amount | Factual error |
| Change court name | Wrong precedential weight |
| Insert unsupported holding | Fabricated ratio |

```python
# data/perturbations/generate_perturbations.py
def generate(source_pairs: Path, out_path: Path, seed: int = 42) -> None:
    """Reads (passage, claim) pairs known to be genuinely entailed, applies one
    perturbation type per output row (tagged with which type), writes
    perturbations.jsonl. Must be releasable standalone -- RESEARCH.md §2.5's hard
    rules require the generator itself to be published, not just its output."""
```

**Hard rules, non-negotiable** (`RESEARCH.md` §2.5, and this is the whole reason the
task lives here rather than inside M8a):
- **Disjoint from every evaluation split.** Tag every perturbation row with its source
  `doc_id`. Write a check (can be a `pytest.skip`-guarded test like M1's pattern, §3
  above, until `data/splits/*.json` exists) that no perturbation's source `doc_id`
  appears in `eval/gold/entailment.jsonl` or any other gold set.
  `tests/test_perturbations.py` should include this check now, skipping with a stated
  reason until splits land, then becoming a real enforced assertion — same convention
  `M1_ONBOARDING.md` §9 established for its own leakage test.
  Once `data/splits/*.json` exists, extend the check to confirm every source `doc_id`
  is drawn from the **training** split specifically — perturbations built from dev/test
  documents would leak model familiarity with those documents into a judge that later
  gets evaluated on them.
- **Never generated from our own system's outputs** — only from genuine IN-Abs
  (passage, claim) pairs. The judge would otherwise learn to like this project's own
  writing style rather than detecting real faithfulness failures.

---

## 10. ClaimRAG-Law's 968 validated claims

`PLAN.md` M6 lists pulling this external set as one of its deliverables. It's used
(by M8a, later) as an out-of-domain generalisation check for InLegalNLI — the paper's
answer to "you built your own test set" (`RESEARCH.md` §2.5's Hard Rules and T11).

Your job in M6 is narrower: locate and cache it (arXiv 2605.21071's associated data
release, if public — check the paper for a data availability statement / repo link
before assuming one exists), store it read-only alongside the other gold sets (e.g.
`eval/gold/external/claimrag_law.jsonl` with a `PROVENANCE.md`-style note on where it
came from and its license), and **do not relabel or modify it** — its value is being
independent of anything built in this project. If it turns out not to be publicly
released, say so explicitly rather than substituting something else silently; this is
exactly the kind of external-dependency surprise `RESEARCH.md` §1's "re-run the
literature search" discipline exists to catch early.

---

## 11. `eval/run_eval.py` and the CI eval-regression gate

```python
def run(split: str, out_dir: Path = Path("eval/results")) -> dict:
    """Runs Pipeline.ask() (unmodified -- see §3) over every item in the named
    gold set(s) relevant to that split, computes every applicable metric from
    eval/metrics.py + eval/stats.py + eval/extractiveness.py, writes a results
    table to eval/results/{split}_{timestamp}.json. Never hand-types a number --
    OPS.md's retention table keeps every eval/results/*.json forever, in git,
    specifically so every number in the eventual paper traces back to one of
    these files."""
```

Right now — with every model still stubbed — this will produce a table of mostly
trivial or near-zero scores. **That's fine and expected.** The done-when criterion
(§2) is that the table *computes end to end*, not that the numbers look good. Getting
this wired now, before any real model exists, is what catches a broken harness at zero
cost instead of after M7's baselines burn real GPU money (`PLAN.md` M7: baselines
exist partly to "validate the harness on real outputs for free... if the harness is
broken you find out at zero training cost" — that logic only works if `run_eval.py`
already runs before M7 starts).

**CI eval-regression gate**: a CI job that runs `run_eval.py --split dev` on every PR
and fails if any metric regresses past a threshold from the last committed
`eval/results/dev_*.json`. Keep the threshold generous for now (the system is all
stubs — you're guarding against the harness silently breaking, not against real
quality regressions that don't exist yet). Tighten it as real components land in later
milestones; that's not your job to predict now.

---

## 12. What already exists — reuse it, don't rebuild it

- **`src/smartlawai/protocols.py`** — `Claim`, `GenerationResult`,
  `VerificationResult` are already defined here (`status: "ok" | "unavailable"`,
  never a silently-substituted score — I2). Your `groundedness.jsonl`/
  `entailment.jsonl` schemas should use compatible field names (`passage_ids`,
  `text`, `citations`) so `run_eval.py` can compare a real `GenerationResult`
  against your gold data without a translation layer.
- **`src/smartlawai/pipeline.py`**'s `Pipeline` class — call `.ask(question, scope)`
  directly from `run_eval.py`. Its own docstring already documents this exact use
  case. Don't reimplement any part of the retrieve/rerank/generate/verify/decide
  sequence inside `eval/`.
- **`src/smartlawai/trace.py`**'s `PipelineTrace.to_dict()` — every `AnswerResult`
  carries a full trace; your results files can and should embed relevant trace detail
  (cost, stage timings) alongside metric scores, not just the bare numbers.
- **`sample_docs/sample_judgment.txt`** — use for structural/smoke testing of
  `run_eval.py` and the annotation app before real gold data exists, the same way
  `M4_ONBOARDING.md` §4 uses it for structural testing only, never for real numbers.
- **`data/raw/summ`** (already pulled by M1) — your source corpus for
  `data/perturbations/`. Don't re-pull it.
- **`pyproject.toml`'s `[data]` extras** already includes `datasets>=2.19,<3.0` and
  `huggingface_hub>=0.23` — useful if you need dataset-loading utilities for
  ClaimRAG-Law (§10) or perturbation source data. `[ui]` already includes
  `streamlit>=1.30` — your annotation app's dependency, don't add a second pin.

---

## 13. Where your code goes — and what NOT to touch

**Build only in these paths**:
- `eval/` at the top level (after the §4 move) — `metrics.py`, `stats.py`,
  `extractiveness.py`, `run_eval.py`, `annotate/`, `gold/`, `results/`
- `data/perturbations/`
- `tests/test_metrics.py` (moved/extended), `tests/test_stats.py`,
  `tests/test_extractiveness.py`, `tests/test_perturbations.py`

**Do not modify**: `gcloud/`, `train/`, `docker/`, `.github/`,
`src/smartlawai/pipeline.py`/`scope.py`/`trace.py`/`protocols.py`/`adapters/*`,
`api/main.py`, `ui/app.py`, anything under `data/` besides the new
`data/perturbations/` directory (M1's territory), and `registry/`/`verify/` (M4/M5's
territory — they don't exist yet; don't create stub versions of them either, since
that risks colliding with the real interface contracts those onboarding docs already
define). If you genuinely need something from one of these, say so explicitly rather
than editing directly.

**If you need a new dependency**, add it to a new `eval` extras group in
`pyproject.toml` (mirroring the `[data]` group's pattern — see `CLAUDE.md`-adjacent
comments in `pyproject.toml` about why extras stay single-purpose):
```toml
eval = [
  "rouge-score>=0.1.2",
  "bert-score>=0.3.13",
  "scipy>=1.11",   # bootstrap resampling / ECE binning
]
```
`bert_score` may already be an implicit dependency of the existing `bertscore_f1`
function — check whether it's installed anywhere already before assuming it needs
adding; if it's missing entirely today, that's worth noting since `tests/test_metrics.py`
doesn't currently exercise `bertscore_f1` at all (no test calls it).

---

## 14. `pyproject.toml`, `.gitignore` changes

Add the `[eval]` extras group above. Full M6 install:
`pip install -e ".[eval,ui,dev]"` (`ui` for the annotation app, `dev` for pytest).

**`.gitignore` additions**: `data/perturbations/perturbations.jsonl` (large,
regenerable from `generate_perturbations.py` + its fixed seed — don't commit the
generated output, only the generator script and its seed).

**Do NOT gitignore**: `eval/gold/*.jsonl` (small, and per I5-adjacent discipline these
are frozen reference data once collected — treat them with the same "commit it, don't
regenerate it silently" care as `data/splits/*.json`), `eval/results/*.json` (OPS.md's
retention table: keep forever, in git), `data/perturbations/generate_perturbations.py`
itself.

---

## 15. Git workflow

```bash
git pull origin main   # get the latest CLAUDE.md/PLAN.md/RESEARCH.md
git checkout -b m6-eval-harness
```

Work on that branch, push it, open a PR against `main` rather than pushing directly —
same convention the rest of this project uses. Since `eval/` (once moved per §4) and
`data/perturbations/` are new or newly-relocated paths, watch specifically for
conflicts with the two test files that currently import the old
`src/smartlawai/eval/metrics.py` location (§4) — those are the one place your work and
existing code genuinely overlap.

Two automated backups already mirror `main` on a delay (`.github/workflows/`) — not
something you need to act on, just know your work is protected once merged.

---

## 16. Setup

```bash
git clone https://github.com/kramjiy/smartlawai.git
cd smartlawai
git checkout -b m6-eval-harness
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[eval,ui,dev]"
pytest -q                        # should already be green: existing M0/M1 test suite
```

---

## 17. Concrete first steps, in order

1. Re-grep the repo for `smartlawai.eval` / `eval.metrics` to confirm the two
   importers named in §4 are still the only ones, then do the move: relocate
   `src/smartlawai/eval/metrics.py` → `eval/metrics.py`, update the two test imports,
   delete the old location.
2. Replace hand-rolled `rouge_l` with the `rouge-score`-backed `rouge_scores()` (§8);
   update its tests; confirm `pytest -q` is green again before moving on.
3. Build `eval/stats.py` (paired bootstrap, ECE, reliability-diagram data,
   Bonferroni correction) and `eval/extractiveness.py`, each with its own test file.
4. Build the annotation app (`eval/annotate/app.py` + `agreement.py`) per §6 — the
   ~4h build `PLAN.md` calls out as the prerequisite for everything else.
5. **Pilot 20 items on one gold set** (pick `groundedness.jsonl` — the most central
   one) before running any set at full volume. Compute agreement, find where the
   rubric is ambiguous, fix it, record items/hour (`RESEARCH.md` §7.3) — that number
   sizes every remaining set's realistic timeline. If it's ugly, that's information to
   report now, not to discover at full scale.
6. Scale up all five gold sets (§7) using the app, applying the correct
   duplication-for-IAA rule per set from the table in §7.
7. Build `data/perturbations/generate_perturbations.py` (§9) against `data/raw/summ`;
   write `tests/test_perturbations.py` with the disjointness check, `pytest.skip`-
   guarded until `data/splits/*.json` exists (same convention as M1's own leakage
   test).
8. Locate and cache ClaimRAG-Law's 968 claims (§10), or document explicitly that it
   isn't publicly available if that turns out to be the case.
9. Build `eval/run_eval.py` (§11); run it against the M0 stub pipeline end to end;
   confirm a results table is produced and written to `eval/results/`.
10. Wire the CI eval-regression gate (§11) with a generous initial threshold.
11. Verify the literal done-when: the full table computes for the untrained system.

---

## 18. Things that are genuinely unknown until you're doing this — don't assume answers

- Whether the top-level `eval/` vs. `src/smartlawai/eval/` move (§4) has any importer
  you haven't found by grep — re-check, don't trust this document's snapshot.
- Whether `bert_score` is actually installed/working anywhere in the current repo —
  no test currently exercises `bertscore_f1`; confirm it works before relying on it.
- Real annotator throughput (items/hour) — entirely unknown until the pilot (§17 step
  5) runs. This number determines whether the ~300–400-item target per gold set is
  realistic on the calendar you have, or needs to be cut and reported honestly.
- Whether ClaimRAG-Law's 968 claims are actually publicly released, and under what
  license (§10) — don't assume; check the paper's data-availability statement.
- Whether recruiting practising advocates (not just students) as a subset of raters
  (`RESEARCH.md` §7.4) is feasible on this project's timeline/budget — this is a
  people/scheduling question to raise explicitly, not something to resolve solo.
- The real anchoring effect size from the 10% cold-vs-assisted comparison (§6) — if
  it's large, that changes how much you can trust the 2–3× throughput assumption
  model-assisted pre-labelling is supposed to buy you.

If any of these turn out to genuinely threaten the milestone (annotator throughput is
too low to hit target sizes, ClaimRAG-Law isn't available, agreement on a set is
unworkably low even after rubric revision), stop and raise it explicitly rather than
pushing through alone or quietly shrinking a gold set without saying so — these affect
the paper's statistical validity, not just code.
