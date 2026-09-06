# Data Provenance — M1 (Data Foundation)

Authoritative record of where the M1 training corpus came from, how it was
built, and what quality checks were applied. Written per `RESEARCH.md` T4
("silver data must be human-validated and its acceptance rate reported
plainly, whatever it turns out to be") and I5 ("document-level splits,
recorded, never regenerated"). This is a freeze/reference document — update
it only when the underlying data actually changes, not to adjust prose.

---

## 1. Source dataset

- **Dataset**: `Exploration-Lab/IL-TUR` (HuggingFace).
- **Revision**: sha `d16219ad0423cc181ec8460d930fd10a907664b6`, last modified
  2025-04-01T12:09:02Z. Recorded at pull time in `data/raw/REVISION.json`.
- **Configs pulled**: `summ` (7,130 docs — Supreme Court judgments with
  professional headnotes, = IN-Abs), `rr` (100 docs — Competition Commission
  of India antitrust matters, rhetorical-role labelled), `lner` (105 docs),
  `lsi` (66k docs). Only `summ` and `rr` are relevant to the section-pair
  construction decision below; `lner`/`lsi` are pulled for other milestones.

## 2. Section-pair construction method: Path 2 (frontier-generated)

**Path 1 (free alignment via `rr`'s rhetorical-role spans) was investigated
first and found infeasible.** `rr` and `summ` share **zero** document ids —
disjoint universes (`rr` is CCI/antitrust, `summ` is Supreme Court judgments).
Full investigation and decision rule in `data/rr_summ_alignment_report.md`.

Two free alternatives were also checked and found insufficient:
- **LegalSeg** (arXiv 2502.05836): gated behind a business-enablement form,
  not pursued.
- **IN-Ext** (Bhattacharya et al., 50 expert-annotated docs): checked by
  actual document content against `summ` — ~0% real overlap (1/50 content
  match, noise-level); these documents were held out of the IN-Abs crawl
  specifically for manual annotation, not part of it.

**Decision**: Path 2 (frontier-LLM-generated section summaries) is required
for the whole `summ` corpus — not a shortcut; every prior group needing
section-aligned legal summarisation data past a few hundred documents hit the
same wall (see `data/rr_summ_alignment_report.md` and `M1_PROGRESS.md` §2 for
literature detail).

## 3. Generation model selection

Five models were compared empirically against our own documents (not an
external leaderboard) via claim-level HHEM scoring, at increasing sample
sizes (n=10 then n=80) with Fisher's exact tests to confirm significance
before committing spend. Full methodology, bugs found, and comparison table
in `M1_PROGRESS.md` §4.

**Winner: Grok 4.1 Fast (Non-Reasoning)** (`xai/grok-4.1-fast-non-reasoning`,
Vertex AI Model Garden, `location="global"`), on cost (~$13.52 for the full
corpus, cheapest of all five candidates tested including Gemini 2.5
Flash-Lite) and quality (n=80 pilot: 96.9% section acceptance, 1.8%
hallucination) simultaneously — confirmed significant via Fisher's exact test
against both runner-ups (p=0.0003 vs Gemini 3.8 Flash, p<0.0001 vs Mistral
Medium 3).

## 4. Production generation run

- **Script**: `data/run_path2_production.py`. Non-reasoning mode, on-demand
  (no batch/offline pricing tier exists for this model — checked directly
  against the live Vertex pricing browser).
- **Result**: 7,130/7,130 source documents processed, 65 errors (0.9%,
  unparseable JSON after 6 retries each), **31,621 section pairs** produced
  → `data/processed/section_pairs_raw.jsonl`. Runtime: 414.9 minutes
  (~6.9 hours), 3 concurrent workers (sized to the verified per-project Grok
  quota: 40 QPM / 10,000 output-TPM).
- Every pair is tagged `provenance: path2-silver` and carries `doc_id`,
  `section` (one of: facts, statute, argument, analysis, judgement),
  `source_headnote` (the original professional summary text used as the
  faithfulness premise), and `generated_text` (the model's output).

## 5. HHEM claim-level filtering (silver data quality gate)

Per `RESEARCH.md`'s own faithfulness definition (claim-level, not
whole-section), every generated section's claims were scored against
premise-chunks of its source headnote using `vectara/
hallucination_evaluation_model`, then the whole section was accepted or
rejected as a unit (not edited claim-by-claim, to keep every retained pair
coherent text rather than a patchwork with sentences excised).

**Method** (`data/score_filter_pairs.py` / `data/pilot/hhem_score_batched.py`):
- Claims = sentence-split `generated_text` (regex on sentence boundaries).
- Premise = `source_headnote`, chunked at 300 words with 50-word overlap
  (`PREMISE_CHUNK_WORDS` / `PREMISE_CHUNK_OVERLAP`) — HHEM has a limited
  context window, so the full headnote can't always be passed as one premise.
- A claim scores as the **max** HHEM score across all premise chunks (the
  correct chunk contains the supporting content; other chunks are
  irrelevant, not evidence of contradiction).
- A claim **passes** at score ≥ 0.5 (HHEM's own "likely faithful" cutoff).
- A section is **accepted** only if ≥ 80% of its claims individually pass
  (`ACCEPT_THRESHOLD = 0.8`). This threshold is a starting point, not yet
  calibrated against human judgment — M5's job is to calibrate real
  thresholds against labelled data (same precedent as `PLAN.md` M5.6's OOS
  threshold); it should be revisited once the 200-item human validation
  sample below exists to check it against.

**Compute**: full-corpus scoring involves every claim checked against every
premise chunk of its own headnote — **1,015,479** (chunk, claim) scoring
operations, not simply the ~212,000 claim count (a wrong early estimate,
corrected once measured). Run as 5 parallel Vertex AI Custom Jobs (4× L4 +
1× A100, proportionally sharded ~1:1:1:1:2.5 by estimated throughput) via
`data/pilot/shard_corpus.py` + `data/pilot/hhem_score_batched.py`. One shard
(A100) hit a CUDA OOM at 88.8% completion on the first attempt (`BATCH_SIZE`
128 padding blew up on a batch containing several long premise chunks);
fixed by reducing to `BATCH_SIZE=32` and adding GCS-backed checkpointing
every 200 batches, then resumed cleanly to completion. Full incident detail
in `M1_PROGRESS.md` §6.

### Final result — the real, non-projected acceptance rate for the whole corpus

| Shard | Pairs | Accepted | Acceptance | Claims | Mean claim score | Hallucination rate |
|---|---|---|---|---|---|---|
| L4-1 | 4,865 | 4,616 | 94.9% | 22,895 | 0.912 | 3.37% |
| L4-2 | 4,865 | 4,698 | 96.6% | 27,771 | 0.933 | 1.85% |
| L4-3 | 4,865 | 4,719 | 97.0% | 30,800 | 0.941 | 1.44% |
| L4-4 | 4,865 | 4,664 | 95.9% | 36,929 | 0.943 | 1.47% |
| A100 | 12,161 | 11,827 | 97.3% | 90,672 | 0.948 | 0.94% |
| **TOTAL** | **31,621** | **30,524** | **96.53%** | **209,067** | **0.9403** | **1.49%** |

(Totals are weighted by claim/pair count, not a simple average across
shards — the A100 shard alone accounts for ~38% of the corpus.)

**This is the number required by `RESEARCH.md` T4, reported plainly: 96.53%
of generated section pairs pass the automated claim-level filter.** It lands
close to (slightly above) the n=80 pilot's 96.9% projection for this same
model, which is evidence the pilot sample was reasonably representative
rather than a favorable draw.

**Output files**:
- `data/processed/section_pairs_scored.jsonl` — all 31,621 pairs, each with
  its full `hhem_claim_scores` array, `hhem_mean_score`, `hhem_pass_rate`,
  and `hhem_accepted` boolean.
- `data/processed/section_pairs_accepted.jsonl` — the 30,524 pairs that
  passed; **this is the corpus that feeds `dedup_split.py` and, downstream,
  M8b fine-tuning.** Rejected pairs are kept in the scored file (not
  deleted) for error analysis and for the human-validation check below.

## 6. Deduplication + document-level split (`data/dedup_split.py`)

**Method**: MinHash near-duplicate detection over `summ`'s full judgment
text (normalized: lowercased, non-alphanumerics stripped, boilerplate
header/citation lines removed — case numbers, bench composition, counsel
names, filing dates — since these generate spurious shingle overlap between
otherwise-unrelated judgments). Word 5-gram shingles, `MinHash(num_perm=128)`,
`MinHashLSH(threshold=0.9)` (the 0.9 threshold is `PLAN.md`'s literal spec).
For each connected component of near-duplicates found, the longest text is
kept (ties broken by lowest numeric doc_id); the rest are dropped.

**Real result**: of 7,130 unique `summ` documents, **29 near-duplicate pairs**
were found (all jaccard estimate 1.0 — exact-content duplicates after
normalization, each its own separate pair, no larger clusters), so **29
documents were dropped**. Full pairwise edges + per-component kept/dropped
detail persisted to `data/processed/near_duplicate_pairs.json` (committed —
this is what `tests/test_data_leakage.py` reads, so CI never recomputes
MinHash).

**Split**: document-level, 80/10/10 over the 7,101 surviving doc_ids, fixed
seed 42 (`random.Random(42).shuffle` on the id list sorted numerically first
for determinism) → **train=5,681, dev=710, test=710** documents. Written to
`data/splits/{train,dev,test}.json` per the schema in `M1_ONBOARDING.md` §7.

**Join**: `section_pairs_accepted.jsonl` (30,524 pairs) joined against the
frozen split by `doc_id`; pairs whose source document was dropped as a
near-duplicate are excluded. The pre-existing `split` field (IL-TUR's own
original train/test split, unrelated to and not to be confused with our own
frozen split) is preserved as `iltur_original_split` and overwritten with our
own split assignment in the `split` field.

**Real result** → `data/processed/section_pairs.jsonl`:

| Split | Section pairs | Documents |
|---|---|---|
| train | 24,346 | 5,681 |
| dev | 3,015 | 710 |
| test | 3,044 | 710 |
| dropped as duplicate | 119 | 29 |
| **Total** | **30,524** | **7,130** |

(24,346 + 3,015 + 3,044 + 119 = 30,524 — accounts for every previously
accepted pair exactly.)

**Performance note, not a data-quality issue**: the first implementation of
`build_minhash()` called `MinHash.update()` once per shingle in a Python
loop; across this corpus's ~28.5M total shingles (one document alone has
117,512) that took 60+ minutes with no clear end. Switched to
`MinHash.update_batch()` (vectorized) and the same computation finished in
under two minutes. No GPU involved or needed — MinHash/LSH is pure
hashing/set-operation work with no matrix math, so cloud GPU compute
would not have helped here at all; this was a batching bug, not a
compute-scale problem.

## 7. What has NOT yet happened (tracked here so it isn't silently skipped)

- **200-item human validation pass** (`RESEARCH.md` T4) — a human-driven
  sample checking whether the HHEM filter's accept/reject judgment agrees
  with a human reader's, stratified across accepted and rejected pairs.
  Not automatable, not yet run. Until this exists, the 96.53% figure in §5
  is the automated filter's self-reported rate, not an independently
  verified one — report it as such.
- **`ACCEPT_THRESHOLD = 0.8` calibration** — currently a placeholder, to be
  checked against the human validation sample once it exists (per M5's
  threshold-calibration precedent).
- Two documents worth a manual read, flagged during pilot testing: `doc=5925`
  failed HHEM across most sections for weaker pilot models (likely a genuine
  content-mismatch case, not scattered noise) — worth checking its status
  (and split assignment) in the final corpus specifically.

## 8. Freeze statement

**Frozen as of 2026-09-06.** `data/splits/{train,dev,test}.json` were
written by `data/dedup_split.py` on this date and are now frozen per I5:
document-level splits, recorded, never regenerated. `tests/test_data_leakage.py`
passes against them (`test_no_doc_id_in_two_splits`,
`test_no_near_duplicate_spans_splits` both green; the eval-leakage test
correctly skips until M6 adds `eval/gold/*.jsonl`).

Any future regeneration of these splits requires both `--force` to
`dedup_split.py` **and** a dated addendum to this section explaining why the
freeze was broken — the script enforces the former, this document is
responsible for the latter.
