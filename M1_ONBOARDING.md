# M1 Onboarding — Data Foundation

You're picking up **M1** on `smartlawai` in a fresh Claude Code chat with no
access to any prior conversation. This doc is everything you need to start
without re-deriving context from anyone. Read it top to bottom once, then use
it as a reference. It is scoped to M1 exclusively — it does not cover M0
(already merged), M4, or anything about GCP/training infrastructure, which
are separate tracks owned elsewhere.

**No GPU needed for any of this.** M1 is pure Python, CPU-only, local
disk. Don't touch Cloud Shell, `gcloud`, or Docker.

---

## 1. Read these three files in the repo root first, in this order

1. **`CLAUDE.md`** — non-negotiable invariants. Read especially **I5**:
   splits are document-level, frozen once written, recorded with provenance,
   and **must never be regenerated**. Also skim §3 (repo layout) and §6
   (`.claudeignore` requirements — that file doesn't exist in the repo yet;
   you'll be creating it).
2. **`PLAN.md`** — find the `## M1` section. That's the literal spec: build
   steps, and the exact "done when" criteria. This doc restates and expands
   it below, but `PLAN.md` is the source of truth if anything here seems to
   drift from it.
3. **`RESEARCH.md`** — read §3 "DATA" in full, and skim §8's threats **T2**
   (leakage), **T4** (silver data), **T5** (OCR ceiling) — M1 is what
   defends the paper against exactly these three threats. Understanding why
   matters as much as the mechanics.

---

## 2. What you're actually building (M1 summary)

Pull IL-TUR's four dataset configs → figure out whether rhetorical-role
labels can be aligned to headnotes to build **gold** section pairs (or
whether a silver, model-generated fallback is needed) → deduplicate near-
identical documents → split document-level into train/dev/test, frozen
forever → write CI tests that make leakage across those splits impossible to
reintroduce silently → pre-tokenize the result for fast downstream loading →
build (but not necessarily finish) an OCR word-error-rate measurement
harness.

**Done when** (verbatim from `PLAN.md`):
- ~35,000 section pairs exist
- splits are frozen with provenance (`data/PROVENANCE.md`)
- leakage tests pass in CI
- the segmentation decision (which path was taken, §5 below) is written down
  with its rationale

The OCR WER number (§10 below) is **not** part of this done-when list —
build the harness, but don't let it block declaring the rest of M1 complete.

---

## 3. The four datasets

All four come from HuggingFace: `Exploration-Lab/IL-TUR`, configs
`"summ"`, `"rr"`, `"lner"`, `"lsi"`.

| Config | Size | Used for |
|---|---|---|
| `summ` (= IN-Abs) | 7,100 SC judgments + professional headnotes | Section-pair summarisation gold — **the primary corpus for M1** |
| `rr` | 21,184 sentences, rhetorical-role labels | Free section segmentation, **if** it aligns to `summ` (§5) |
| `lsi` | 66k docs, statute identification | Cached now for **M4**'s registry stress test — NOT deduped/split/pre-tokenized in M1 |
| `lner` | 105 docs | Cached now for a later NER extension — NOT deduped/split/pre-tokenized in M1 |

Only `summ`/`rr` feed the dedup → split → pre-tokenize pipeline. `lsi` and
`lner` are pulled and cached in M1 (per `PLAN.md`'s step 1) purely so a later
milestone doesn't need to re-pull them, but deduping/splitting them now would
burn real time for zero M1 benefit — don't do it.

---

## 4. Directory and file layout

Mirrors the existing `eval/` pattern already in this repo (code and data
colocated under one topic directory — `eval/metrics.py` sits next to
`eval/gold/` today, even though the top-level `eval/` dir doesn't exist yet
either; see note in §9).

```
data/
├── prepare_iltur.py          # step: pull + cache all 4 IL-TUR configs
├── align_rr_summ.py          # step: RR<->SUMM alignment investigation (read-only report)
├── build_pairs.py            # step: materializes section pairs (Path 1 and/or Path 2)
├── dedup_split.py            # step: MinHash dedup + doc-level split (--force guarded)
├── pretokenize.py            # step: Arrow dataset build
├── raw/                      # GITIGNORED -- load_dataset()/save_to_disk() cache
├── processed/                # GITIGNORED -- regenerable intermediate + final artifacts
│   ├── section_pairs_raw.jsonl      # pre-dedup, pre-split, all docs, provenance-tagged
│   ├── section_pairs.jsonl          # post-dedup, post-split, has a `split` column
│   ├── near_duplicate_pairs.json    # MinHash/near-dup graph -- COMMITTED (small, feeds CI)
│   └── section_pairs.arrow/         # datasets.Dataset.save_to_disk() output
├── ocr_wer_gold/              # HUMAN-SUPPLIED, GITIGNORED (pages/), see §10
│   ├── pages/{page_id}.{png|pdf}
│   └── transcripts/{page_id}.txt
├── splits/                   # COMMITTED, frozen (I5)
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── rr_summ_alignment_report.md   # COMMITTED -- output of the §5 investigation
└── PROVENANCE.md              # COMMITTED

eval/
├── ocr_wer.py                 # harness (new top-level eval/ dir, see §9 note)
└── ocr_wer.md                 # COMMITTED generated report

tests/
└── test_data_leakage.py       # flat file, follows existing tests/ convention
```

Note: `data/processed/near_duplicate_pairs.json` is committed even though the
rest of `data/processed/` is gitignored — it's small (a list of doc-id
pairs, not raw text) and committing it means the CI leakage tests never need
`data/raw/` or `data/processed/`'s larger files present to run.

---

## 5. The IL-TUR pull — `data/prepare_iltur.py`

```python
CONFIGS = ["summ", "rr", "lner", "lsi"]

def pull_config(config_name: str, cache_dir: Path = Path("data/raw"),
                 force: bool = False) -> Path:
    """load_dataset(...) then save_to_disk() so re-runs don't re-download.
    Idempotent: skips if the target dir already has a valid dataset."""

def main(configs: list[str], force: bool = False) -> None: ...  # argparse CLI
```

- Exact calls: `load_dataset("Exploration-Lab/IL-TUR", "summ", cache_dir="data/raw")`,
  identically for `"rr"`, `"lner"`, `"lsi"`.
- **Pull `lner` first** (105 docs, smallest) as a connectivity + schema smoke
  test before spending time/bandwidth on the larger three — in particular
  check `lsi`'s real download size before committing to a full pull; it's
  the one config that could turn out to be unexpectedly large.
- Each config's result is written via `.save_to_disk("data/raw/<config>")` —
  a `DatasetDict`, memory-mapped Arrow, independent of the `datasets`
  library's own cache format across versions.
- Capture the resolved dataset revision/sha (via
  `huggingface_hub.HfApi().dataset_info("Exploration-Lab/IL-TUR").sha`) and
  record it — this goes into `data/PROVENANCE.md` (§8).
- Do the minimum here: no filtering, no cleaning. Just cache-and-freeze the
  raw pull. All real processing happens in `align_rr_summ.py`/`build_pairs.py`.

---

## 6. RR → SUMM alignment investigation — `data/align_rr_summ.py`

**This is a hard checkpoint.** Nothing past it (dedup, split, leakage tests,
pre-tokenization) can proceed until this decision is made, because it
determines what a "section pair" even is. The outcome is genuinely unknown
until you've pulled the real data — do not assume an answer.

What the investigation script needs to establish, in order:

1. **Print both configs' features** (`ds["rr"].features`, `ds["summ"].features`)
   to find the real column names. Do not assume field names from any prior
   description — verify against the actual downloaded data.
2. **Check for a shared, joinable document identifier** between `rr` and
   `summ`. If the two configs draw from disjoint document universes, Path 1
   below is dead immediately, regardless of alignment quality.
3. **For each doc with a shared id**: segment `rr`'s per-sentence rhetorical-
   role sequence into contiguous same-label spans (e.g. `FACTS`, `RULING`,
   `ARGUMENT`, ...). Test whether these spans correspond to structural breaks
   in `summ`'s headnote text (the headnote may already be segmented into
   sections, or may be one blob needing a heuristic split — check the real
   field before assuming either).
4. **Define "clean alignment"** per document as: at least 3 contiguous
   RR-labelled spans found, each with a non-trivial correspondence to a
   headnote section boundary (sentence-order correlation or lexical-overlap
   — tune the exact metric against the real data you're looking at; this
   number isn't fixed in advance because it depends on what the data
   actually looks like).
5. **Compute corpus-level coverage** = (docs with clean alignment) / (docs
   with a shared id in both configs).

### Decision rule

- **Coverage ≥ 70%** → **Path 1 primary**. Use RR-aligned gold pairs for
  eligible docs; fall back to Path 2 (below) only for the uncovered
  remainder, tagged accordingly.
- **Coverage < 70%** → **Path 2 primary** for the whole corpus. Keep any
  docs that did pass clean alignment as a bonus gold subset rather than
  discarding them.

This is a coverage threshold, not an all-or-nothing switch, because
`RESEARCH.md`'s T4 mitigation is about *never concealing* which pairs are
silver — a hybrid corpus with an explicit per-pair provenance tag satisfies
that just as well as a pure-Path-1 corpus, and throwing away genuinely clean
RR-aligned pairs because overall coverage is imperfect would waste real gold
data for no reason.

**Output**: `data/rr_summ_alignment_report.md` — the coverage number, which
rule fired, a handful of example aligned and misaligned docs. This becomes
the primary source for the "segmentation decision, with rationale" that
`PLAN.md`'s done-when criteria requires.

### Path 2 — the fallback

Frontier-LLM-generated section summaries, then a **200-item human
validation pass**, with the **acceptance rate reported plainly** (never
omitted, even if low — that's the whole point of T4's mitigation).

**Cost warning, read this before running anything in Path 2 at scale:**
generating section summaries for up to ~7,100 documents via a frontier
model API has a real dollar cost that has **zero line item** in `PLAN.md`'s
`COMPUTE BUDGET` table. Before running a real generation pass:

1. Build a `--dry-run`/cost-estimate mode into `build_pairs.py` that reports
   estimated token volume and dollar cost for the model you intend to use.
2. **Surface that estimate to the user and get explicit go-ahead before
   spending anything.** This mirrors the "smoke test before the real run"
   discipline already used elsewhere in this project for billed GPU jobs —
   the same principle applies here even though the billing surface is an
   LLM API rather than GCP.

---

## 7. Dedup + document-level split — `data/dedup_split.py`

- **Library**: `datasketch` (`MinHash` + `MinHashLSH`) — pure Python,
  lightweight, no heavy deps.
- **Computed over**: normalized full judgment text from `summ` (lowercased,
  whitespace-collapsed, citation/header noise stripped) — **not** `lsi`'s
  66k docs, which aren't part of this pipeline (§3).
- Shingle on word 5-grams; `MinHash(num_perm=128)`;
  `MinHashLSH(threshold=0.9)` — the 0.9 threshold is `PLAN.md`'s literal
  spec, don't change it without updating that file too.
- For each connected component of near-duplicates found: keep exactly one
  canonical doc (longest text; ties broken by earliest doc_id), drop the
  rest. Record the dropped count in `PROVENANCE.md`.
- **Persist the near-dup pairs graph** to
  `data/processed/near_duplicate_pairs.json` regardless of what dedup action
  was taken — this file is committed (see §4) and is what the CI leakage
  test (§9) reads, so CI never needs to recompute MinHash from scratch.
- **Split**: document-level, **80/10/10** train/dev/test via a fixed-seed
  shuffle of surviving doc_ids (`PLAN.md` doesn't specify a ratio; with
  ~7,100 `summ` docs this gives roughly 5,680/710/710 — comfortably above
  M6's ~200-300-item gold-set sizing needs on the dev/test side, while
  keeping the training pool as large as possible since M8b's LoRA training
  volume is the scarcer resource here).
- **Split JSON schema** (`data/splits/{train,dev,test}.json`):
  ```json
  {
    "split": "train",
    "source": "IL-TUR/summ",
    "doc_ids": ["...", "..."],
    "count": 5680,
    "dedup_threshold": 0.9,
    "seed": 42,
    "frozen_at": "2026-XX-XX"
  }
  ```
  This doc-id list is the minimum I5 requires. Downstream consumers
  (M2's retrieval-corpus build, M8's LoRA-training-data selection) join
  against `data/processed/section_pairs.jsonl`'s own `doc_id`/`split`
  columns — nothing richer needs to live in the split files themselves.
- **`--force` safety behavior**: on run, the script checks whether
  `data/splits/train.json` (etc.) already exist. If so, **it refuses and
  exits with an error** naming I5 and instructing that regeneration
  requires both `--force` *and* a manual addendum to `data/PROVENANCE.md`
  explaining why the freeze was broken. `--force` alone only lifts the
  file-exists guard — it does not skip the manual documentation step. This
  is I5 enforced by the tooling, not left as a comment someone might ignore.

---

## 8. `data/PROVENANCE.md` — required contents

- IL-TUR dataset revision/sha pulled (from §5), date pulled, the exact
  `load_dataset` calls used.
- Section-pair construction path taken (Path 1 / hybrid / Path 2) **with
  the coverage number and rationale** from `align_rr_summ.py`'s report —
  this is the literal "segmentation decision, with rationale" `PLAN.md`
  requires.
- If Path 2 was used at all: the 200-item human-validation acceptance rate,
  reported plainly, never omitted even if low.
- MinHash parameters (shingle size, `num_perm`, LSH threshold), count of
  near-duplicate docs dropped and which were kept.
- Split ratios, seed, and final per-split doc counts and section-pair counts.
- Freeze date, and an explicit statement: *"These splits are frozen per I5
  and must never be regenerated without a documented, approved exception
  recorded here."*

---

## 9. Leakage CI tests — `tests/test_data_leakage.py`

Three tests, following the existing flat-file `tests/` convention (see
`tests/test_scope.py` for the style: plain functions, no fixtures needed
unless state is shared):

- **`test_no_doc_id_in_two_splits()`** — load all three
  `data/splits/*.json`, assert pairwise `set(doc_ids)` intersections are
  empty.
- **`test_no_near_duplicate_spans_splits()`** — load the committed
  `data/processed/near_duplicate_pairs.json` plus the three split files;
  for every near-dup pair recorded, assert both members are never in
  different splits.
- **`test_no_qa_eval_item_from_training_doc()`** — implemented as a reusable
  helper this test calls, e.g.
  `assert_no_eval_leakage(eval_glob="eval/gold/*.jsonl", doc_id_field="source_doc_id", train_split="data/splits/train.json")`.
  Since `eval/gold/*.jsonl` doesn't exist until **M6**, this test globs for
  matching files and calls `pytest.skip("no eval/gold/*.jsonl yet -- M6 not landed")`
  when none are found. That makes it a real, enforced assertion the moment
  M6 adds gold files, with zero further code changes needed then — not a
  vacuous pass, an honest skip with a stated reason.

Note on the top-level `eval/` directory: it doesn't exist in the repo yet
(only `src/smartlawai/eval/metrics.py` does, a pre-existing inconsistency
from the original v1 import). M1 creates `eval/` at the top level because
`PLAN.md` names `eval/ocr_wer.md` as a literal output path (§10). **Create
only `eval/ocr_wer.py` and `eval/ocr_wer.md` there — do not move or touch
`src/smartlawai/eval/metrics.py`**; that inconsistency is out of scope for
M1 and belongs to whoever eventually does the M6 eval-harness work.

---

## 10. Pre-tokenization — `data/pretokenize.py`

Precompute **InLegalBERT** token ids as additional columns alongside the
plain text columns, in a single `datasets.Dataset.save_to_disk()` output at
`data/processed/section_pairs.arrow/`.

Why InLegalBERT specifically: it's the one encoder reused across three later
milestones — M2 (retrieval), M5 (verification), M8a (encoder fine-tuning)
(see `src/smartlawai/core/inlegalbert.py` and `RESEARCH.md` §2.5:
"retrieval, reranking, and verification all InLegalBERT"). Precomputing its
tokenization once here means those three milestones don't each redundantly
re-tokenize the same text with the same tokenizer. It does **not** overreach
into M8b's territory — QLoRA's per-arm tokenizers (Qwen2.5-3B/7B/14B,
Mistral-7B) are model-specific and stay entirely M8b's job; nothing here
assumes or hardcodes those.

- Tokenizer: `AutoTokenizer.from_pretrained("law-ai/InLegalBERT")`,
  `max_len=512` (matches `core/inlegalbert.py`'s existing default),
  truncate/pad at this stage.
- Schema: `doc_id, split, section_id, source_text, target_text, provenance,
  source_input_ids, source_attention_mask, target_input_ids, target_attention_mask`.
- Reads from `data/processed/section_pairs.jsonl` — **must run after** §7
  (dedup + split), so the `split` column is final and frozen before
  tokenization happens.
- Needs both the `data` and `ml` extras installed (§12) — the tokenizer
  comes from `transformers`, which lives under `[ml]`, not `[data]`.

---

## 11. OCR WER harness — `eval/ocr_wer.py`

```python
def compute_wer(reference: str, hypothesis: str) -> float: ...   # via jiwer
def run_ocr_wer_eval(gold_dir: Path = Path("data/ocr_wer_gold"),
                      out_path: Path = Path("eval/ocr_wer.md")) -> None: ...
```

**Where Claude's responsibility ends and a human deliverable begins — do not
blur this line.** This step needs 50 hand-transcribed scanned pages, which
is not something an agent can produce. Your job is the harness and the
report generation only.

- Expected human-supplied structure: `data/ocr_wer_gold/pages/{page_id}.{png|pdf}`
  (scanned source pages) paired 1:1 with
  `data/ocr_wer_gold/transcripts/{page_id}.txt` (hand transcription), 50
  matched pairs total.
- **If the directory is missing or has fewer than 50 matched pairs, the
  script must hard-fail** (`SystemExit`) with an explicit message naming
  the required structure and count. **Never fabricate, approximate, or
  silently sample fewer than 50 pages.**
- For each page, reuse `src/smartlawai/core/ocr.py`'s existing extraction
  path (the `_ocr_pdf`/pytesseract branch inside `extract_text`) to produce
  the OCR hypothesis — don't reimplement OCR calls from scratch; a scanned
  page naturally routes through the OCR branch already in that module.
- Compute WER per page via `jiwer` (new dependency, §12), aggregate
  mean/median, write `eval/ocr_wer.md` as a markdown table + summary stats
  + gold-set size + date.
- **This is not a blocker for the rest of M1.** `PLAN.md`'s literal M1
  done-when list (§2 above) doesn't include the WER number. Build the
  harness now regardless, since `RESEARCH.md`'s T5 says OCR quality "caps
  everything downstream" and you want the measurement as soon as the human
  transcriptions exist — but don't hold up declaring the rest of M1 done
  while waiting on that human work to land.

**Open question, don't decide unilaterally**: should
`data/ocr_wer_gold/transcripts/` (the 50 human transcriptions themselves —
small text files) be committed to the repo for reproducibility of the WER
number, or gitignored alongside the page images for consistency/privacy?
Ask before choosing.

---

## 12. `pyproject.toml` changes

New `data` extras group, deliberately kept **separate** from `[train]`:

```toml
data = [
  "datasets>=2.19,<3.0",   # same pin as [train] -- avoid two divergent pins of one package
  "huggingface_hub>=0.23",
  "datasketch>=1.6",
  "jiwer>=3.0",
]
```

Why not fold into `[train]`: `[train]`'s pins (`peft`, `trl`, `bitsandbytes`,
`accelerate`, `torch`) are tightly version-coupled to the GPU/QLoRA stack
and already documented in that file's own comments as fragile ("breaks
constantly"). M1's needs are lightweight, CPU-only, and used by a
completely different workflow much earlier in the project. Bundling them
would force anyone doing data-prep work to install `bitsandbytes`/`torch`
for no reason, and vice versa.

`pretokenize.py` (§10) additionally needs `transformers` from the existing
`[ml]` group — don't add a second `transformers` pin under `[data]`. Full
M1 install: `pip install -e ".[data,ml]"`.

---

## 13. `.gitignore` / `.claudeignore` changes

**`.gitignore` additions**: `data/raw/`, `data/processed/`,
`data/ocr_wer_gold/pages/`. All three are large and/or fully regenerable
from the scripts plus the frozen splits (the page images are also
potentially copyright-sensitive scanned material).

**Do NOT gitignore**: `data/splits/*.json`, `data/PROVENANCE.md`,
`data/rr_summ_alignment_report.md`, `data/processed/near_duplicate_pairs.json`,
`eval/ocr_wer.md` — these are small and load-bearing; I5 requires the
frozen splits to be recorded in git, not merely reproducible from a script.

**`.claudeignore` doesn't exist in the repo at all yet**, despite
`CLAUDE.md` §6 saying it "must contain" a specific list of entries. Create
it now — M1 is what makes this pre-existing gap concrete, since it
introduces the first genuinely large data directories. Populate with
`CLAUDE.md`'s listed entries plus the two new large M1 directories:

```
__pycache__/
*.ipynb
.smartlaw_local/
artifacts/
models/
node_modules/
data/raw/
*.duckdb
*.bin
data/processed/
```

---

## 14. What already exists — reuse it, don't rebuild it

- `src/smartlawai/core/ocr.py` — `extract_text()`, and specifically its
  `_ocr_pdf`/pytesseract branch — reuse this for the OCR WER harness (§11)
  rather than writing new OCR-calling code.
- `src/smartlawai/core/inlegalbert.py` — the InLegalBERT encoder wrapper;
  `pretokenize.py` (§10) should match its `max_len=512` default so
  tokenization stays consistent with how the encoder will actually be used
  in M2/M5/M8a.
- `pyproject.toml`'s existing `[train]` extras already pins
  `datasets>=2.19,<3.0` — match that exact pin in the new `[data]` group
  (§12) rather than choosing a different one.
- `tests/` convention — flat files, no subdirectories, plain functions
  (see `tests/test_scope.py`, `tests/test_local_backend.py`), `conftest.py`
  at the repo root puts `src/` on `sys.path`. Follow this exactly for
  `tests/test_data_leakage.py`.

## Where your code goes — and what NOT to touch

**Build only in these paths** (all new, none exist yet except where noted):
`data/*.py` (the five scripts above), `data/raw/`, `data/processed/`,
`data/splits/`, `data/ocr_wer_gold/` (human-populated), `data/PROVENANCE.md`,
`data/rr_summ_alignment_report.md`, `eval/ocr_wer.py`, `eval/ocr_wer.md`,
`tests/test_data_leakage.py`, plus the `pyproject.toml`/`.gitignore`/
`.claudeignore` edits described above.

**Do not modify**: `gcloud/`, `train/`, `docker/`, `.github/`,
`src/smartlawai/pipeline.py`/`scope.py`/`trace.py`/`protocols.py`/
`adapters/*`, `api/main.py`, `ui/app.py`, or `src/smartlawai/eval/metrics.py`
(the pre-existing inconsistency noted in §9 — not M1's to fix). None of
these are M1's concern; M1 produces data artifacts and data-prep code only,
which later milestones (M2, M6, M8) will read but M1 itself never wires into
the running pipeline.

---

## 15. Git workflow

```bash
git pull origin main   # get the latest CLAUDE.md/PLAN.md
git checkout -b m1-data-foundation
```

Work on that branch, push it, open a PR against `main` rather than pushing
directly — same pattern the rest of this project uses. Since everything M1
touches (`data/`, `eval/ocr_wer.*`, `tests/test_data_leakage.py`, and small
edits to `pyproject.toml`/`.gitignore`) is either new or additive, merge
conflicts with any other in-flight work should be minimal.

## 16. Setup

```bash
git clone https://github.com/kramjiy/smartlawai.git
cd smartlawai
git checkout -b m1-data-foundation
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[data,ml,dev]"
pytest -q                        # should already be green -- the M0 suite
```

---

## 17. Concrete first steps, in order

1. Add the `[data]` extras group to `pyproject.toml` (§12); `pip install -e ".[data]"`.
2. **Connectivity smoke test**: run `data/prepare_iltur.py` for `lner` only
   (smallest config) — confirms HF Hub is reachable from wherever you're
   running this and that the schema is inspectable, before committing to
   the larger pulls.
3. Pull all four configs to `data/raw/`; record the resolved revision/sha.
4. **Hard checkpoint** — run the RR→SUMM alignment investigation (§6)
   against the real downloaded `rr`/`summ` data; compute coverage; apply
   the decision rule. **Do not proceed past this step until the decision
   is made and written to `data/rr_summ_alignment_report.md`.**
   - If Path 2 is triggered at meaningful scale: stop and surface the
     estimated frontier-API cost to the user before spending anything (§6).
5. Materialize `data/processed/section_pairs_raw.jsonl` per the chosen
   path, each pair tagged with its `provenance`. (If Path 2 is used: also
   run the 200-item human-validation pass and record the acceptance rate.)
6. Run `data/dedup_split.py`: MinHash dedup over `summ` doc text → persist
   `near_duplicate_pairs.json` → 80/10/10 doc-level split → write
   `data/splits/{train,dev,test}.json` (refuses to overwrite without
   `--force`, per §7).
7. Join the raw pairs against the final splits →
   `data/processed/section_pairs.jsonl` with a `split` column → write the
   final `data/PROVENANCE.md` (§8). **This is the freeze point** — after
   this, the splits must never be regenerated without a documented exception.
8. Add `tests/test_data_leakage.py` (§9); `pytest -q` green.
9. Run `data/pretokenize.py` to build `data/processed/section_pairs.arrow/`
   (needs the `ml` extra too).
10. Build `eval/ocr_wer.py` (§11). Run it once a human has populated
    `data/ocr_wer_gold/` with 50 page/transcript pairs — this can trail the
    rest of M1 on the calendar since it isn't part of the literal done-when.
11. Verify the literal done-when: ~35k section pairs exist, splits are
    frozen and committed with provenance, leakage tests are green, and the
    segmentation decision is documented with its rationale.

---

## 18. Things that are genuinely unknown until you're doing this — don't assume answers

- Whether `rr` and `summ` share a joinable document identifier at all.
- Whether RR's rhetorical-role span boundaries actually correspond to
  SUMM's headnote section breaks with usable quality — the entire Path 1
  vs. Path 2 decision hinges on this and cannot be predicted in advance.
- `lsi`'s real download size (66k docs — size it before committing to a
  full pull).
- Whether HF Hub is reachable from wherever you're running this (verify in
  step 2 of §17, don't assume).
- The exact column/schema names in each IL-TUR config — verify against the
  real data, not any description written before it was pulled.
- Whether ~35,000 section pairs is actually achievable once real data is in
  hand — depends on how many sections-per-document the headnotes actually
  segment into.
- Path 2's real dollar cost, if triggered — depends on which frontier model
  you use, token volume, and pricing at the time you run it.

If any of these turn out to genuinely threaten the milestone (e.g., `lsi` is
huge, or coverage lands in an ambiguous middle zone, or Path 2's cost is
large), stop and raise it explicitly rather than pushing through alone or
quietly lowering the bar — these affect the paper's argument, not just code.
