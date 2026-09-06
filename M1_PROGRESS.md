# M1 Progress Log — Data Foundation

Running record of everything implemented, decided, and learned during M1's
execution. Companion to `M1_ONBOARDING.md` (the pre-work spec) — this is the
"what actually happened" doc. Updated as work continues; treat the most
recent section as current status.

---

## 1. Research-positioning upgrade (done before M1's technical work started)

Grounded in a live literature scan (9 web searches/fetches: arXiv, ACL
Anthology, HuggingFace) to strengthen the paper's originality/NAACL-fit.

**Changes made to `RESEARCH.md`**:
- New "Framed as a general NLP problem" subsection — repositions the
  authority registry as an instance of *temporal knowledge grounding*, an
  active NAACL/ACL/EMNLP subfield (cites *When Facts Change* ACL 2026,
  *DynamicQA*, *Temporal Validity in Retrieval Memory*).
- Extended related-work table: ClaimRAG-Law (open question flagged, not
  asserted — need to read full text before finalizing the "entailment can't
  catch this" claim), *Who Checks the Citations?* (Princeton, fabrication not
  staleness — orthogonal), *Citation Grounding via Legal Citation Graphs*
  (their own reviewers hit our exact gap and said so — quotable), Stanford
  JELS 2025 (17–33% hallucination in commercial legal AI — motivating stat),
  Falkor-IRAC, Domain-Partitioned Hybrid RAG (both no in-force/registry
  mechanism; latter uses LLM-as-judge, which `I3` rejects).
- New formal, domain-agnostic definition: **authority-grounded (temporal)
  faithfulness** — entailment-based faithfulness checks "is claim supported by
  passage"; this adds "was the cited authority in force as of date d."
- New `## 12. V2 — DEFERRED EXTENSIONS` section (research rationale side).

**Changes made to `PLAN.md`**:
- M7: added a citation-existence-only baseline (replicates the Citation
  Grounding mechanism at small scale on our corpus).
- Pitfall checklist: "novelty claim stale at submission" row now anchored to
  this session's Aug 2026 scan as a dated starting point.
- M4 schema: noted it's already repeal-event-agnostic (no migration needed
  for a second natural experiment later).
- M6: annotation app/gold-set schema noted as annotator-pool-agnostic and
  release-ready (supports the V2 "expert annotation" and "publish as
  standalone benchmark" extensions without rework).
- New `## V2 — DEFERRED EXTENSIONS` section (engineering rationale side):
  second natural experiment, standalone published diagnostic benchmark,
  expert-annotated IndoLexQA, multilingual verification (flagged as *not* a
  cheap drop-in, unlike the other three).

No code, no cost — documentation only.

---

## 2. M1 data pipeline: the hard checkpoint and its dead end

**Files built**: `data/prepare_iltur.py`, `data/align_rr_summ.py`,
`data/build_pairs.py`.

**Pulled from HuggingFace** (`Exploration-Lab/IL-TUR`, revision recorded in
`data/raw/REVISION.json`): `summ` (7,130 docs, = IN-Abs), `rr` (100 docs),
`lner` (105 docs), `lsi` (66k docs, ~232MB).

**Hard checkpoint result — Path 1 is dead**: `rr` and `summ` share **zero**
document ids. `rr` draws from Competition Commission of India antitrust
matters; `summ` from Supreme Court judgments — completely disjoint corpora,
confirmed both by id and by domain. Written to
`data/rr_summ_alignment_report.md`.

**Free alternatives checked, also dead ends**:
- **LegalSeg** (arXiv 2502.05836): gated on Vertex/HF, required a business-
  style enablement form partway through — not pursued further given Path 2
  was already moving forward by the time this was checked.
- **IN-Ext** (Bhattacharya et al., 50 docs, genuine expert-annotated
  segment-wise summaries): checked by actual document content (not just
  filename) against `summ` — **~0% real overlap** (1/50 content match, noise-
  level). These 50 documents were apparently held out of the main IN-Abs
  crawl specifically for manual annotation, not part of it.

**Conclusion, backed by the literature** (not just this project's own
finding): every prior group that needed rhetorical-role/section-aligned data
past a few hundred documents (Kalamkar et al. 354 docs, LegalSeg's ~7,120 via
a *trained classifier*, not manual labels) hit the same wall. **Path 2
(frontier-model-generated section summaries) is required**, matching
standard practice in this specific research area, not a shortcut.

---

## 3. GCP / Vertex AI infrastructure

- **gcloud CLI** installed via `winget install Google.CloudSDK`.
- **Authenticated**: `gcloud auth login` + `gcloud auth application-default
  login` as `m9148230255@gmail.com`. Project: `smartlawai-1` (billing account
  `01DD7B-EAD07F-D571F5`).
- **Cost safeguard**: GCP Billing Budget `d8e04ec5-396b-4264-a1c5-cb80e6c427c1`
  ("SmartLawAI M1 cost safeguard"), ₹20,000 ceiling, 20 threshold rules at 5%
  steps (= ₹1,000 increments), email alerts to the billing admin account.
- **GCS bucket**: `gs://smartlawai-1-m1-pilot/` — staging area for all pilot
  and production data interchange with Vertex Custom Jobs.
- **Quota corrections learned the hard way** (see §6) — always verify via the
  live Console Quotas page, not a model card or a remembered number:
  - Compute Engine raw VM creation is blocked by a separate `GPUS_ALL_REGIONS`
    quota (0 by default) — **irrelevant** once we switched to Vertex AI
    Custom Jobs, which draw from a different quota pool ("Agent Platform API"
    / "Custom model training Nvidia L4/A100 GPUs per region", approved at
    3× L4 and 1× A100 per region for `us-central1` and `asia-southeast1`).
  - Grok 4.1 Fast Non-Reasoning's real per-project quota (verified via
    Console, not the model card): **40 QPM / 220,000 input-TPM / 10,000
    output-TPM** — output-TPM was the binding constraint on generation
    throughput.
  - A quota increase request (Output-TPM/Input-TPM) was **denied** — new
    project/billing account, no history yet. Message suggested waiting 48h.
    Proceeded without it rather than waiting.

---

## 4. Model selection: 5-model pilot, methodology and final result

**Why a model at all is needed** (not a shortcut, see §2): Path 2 needs to
match judgment sections to the *existing* professional headnote — a reading-
comprehension-and-alignment task no classifier or free dataset solves.

**Methodology evolution**:
1. Started with a general external benchmark (Vectara Hallucination
   Leaderboard) to pick candidates — useful for initial triage, but the user
   correctly pushed to stop relying on it and measure faithfulness on *our
   own* documents instead once real testing was feasible.
2. Built `data/score_filter_pairs.py` / `data/pilot/hhem_score_vertex.py`:
   **claim-level** HHEM scoring (RESEARCH.md's own definition — scoring a
   whole multi-sentence section as one hypothesis understates faithfulness,
   since a synthesized section draws on content scattered across the source).
   Runs on a single L4 via Vertex AI Custom Jobs (proven pattern reused
   throughout).
3. First comparison at **n=10** documents — found Gemini 3.8 Flash
   dramatically better than Gemini 2.5 Flash-Lite (93.6% vs 65.2% section
   acceptance). But Fisher's exact test showed the top 3 candidates
   (Mistral/3.8-Flash/Grok 4.20) were **statistically indistinguishable**
   at this sample size (p=1.0).
4. Re-ran the top 2 (Mistral, Gemini 3.8 Flash) at **n=80** (parallelized —
   generation runs on the provider's hosted infra, not local GPU, so
   concurrency is free) — this time found a real, near-significant gap
   (p=0.090) favoring Gemini 3.8 Flash.
5. Added **Grok 4.1 Fast (Non-Reasoning)** at n=80 on the user's request —
   won decisively: **p=0.0003 vs Gemini 3.8 Flash, p<0.0001 vs Mistral**,
   real statistical power this time, not a tie.

**Final comparison table** (n=80 except where noted; n=10 not re-tested for
Gemini 3.1 Pro / Grok 4.20 Reasoning since they were already dominated):

| Model | Cost (full corpus) | Mean claim score | Hallucination | Section acceptance |
|---|---|---|---|---|
| Gemini 2.5 Flash-Lite (n=10) | ~$4.65 (batch) | 0.727 | 25.4% | 65.2% |
| Mistral Medium 3 (n=80) | ~$16.12 (batch) | 0.895 | 7.5% | 85.9% (330/384) |
| Gemini 3.8 Flash (n=80) | ~$30.02 (batch) | 0.915 | 4.0% | 90.1% (319/354) |
| Gemini 3.1 Pro Preview (n=10) | ~$91.09 (batch) | 0.936 | 2.0% | 89.4% (dominated) |
| Grok 4.20 Reasoning (n=10) | ~$85.96 (batch, no batch tier really) | 0.942 | 1.1% | 95.7% |
| **Grok 4.1 Fast Non-Reasoning (n=80, WINNER)** | **~$13.52 (no batch tier)** | **0.934** | **1.8%** | **96.9% (342/353)** |

**Real bugs found and fixed along the way** (worth remembering for future
model integration work):
- Mistral Medium 3's early 403s were from malformed hand-rolled REST calls,
  not access issues — fixed via `aiplatform_v1.PredictionServiceClient.
  raw_predict()` with `httpbody_pb2.HttpBody`.
- Grok models needed the **OpenAI-compatible endpoint at `location="global"`**,
  not `raw_predict` — a completely different failure mode (400 "OpenMaaS
  model is not allowed to be called from this method") than Mistral's.
- The entire **Gemini 3.x family** (3.7, 3.8, 3.1 Pro) is only served via
  `location="global"`, not regional endpoints — every earlier 404 was this,
  not an enablement/quota/rollout problem. Gemini 2.5-series works on any
  region.
- Reasoning-mode models (Grok 4.20 Reasoning) burn large hidden
  "reasoning_tokens" even for simple prompts (~5,085/doc average) — a real,
  easy-to-miss cost multiplier not visible in the response content itself.
- Some models (Gemini 3.1 Pro, 1/10 docs) occasionally wrap valid JSON in a
  list `[{...}]` instead of the requested dict, despite
  `response_mime_type="application/json"` — handled defensively.
- **Not everything with a "Serverless" label in Model Garden is actually
  callable** — DeepSeek-V3.2 and Qwen3-235B both showed as "Serverless" in
  the console but 404'd on every calling pattern tried (self-deploy only in
  practice for this project). Trust empirical test results over the console
  label.

**Cost-safety lessons**:
- Concurrency does not change total cost — only wall-clock time. Total spend
  is tokens × price, independent of worker count.
- A "burst" test (e.g., 20-40 requests) can look clean even when a real
  per-minute quota exists, if the burst doesn't generate enough volume to
  hit it — don't extrapolate short-burst success to sustained-load safety.
- No batch/offline pricing tier exists for Grok 4.1 Fast Non-Reasoning
  (checked directly against the live Vertex pricing browser) — real cost is
  the plain on-demand rate (~$13.52), not a batch-discounted one.

---

## 5. Production Path 2 generation — COMPLETE

**Script**: `data/run_path2_production.py`. Model: Grok 4.1 Fast
(Non-Reasoning), on-demand (no batch tier), `location="global"`.

**Unattended-overnight-run hardening** (real risks raised by the user, both
fixed):
- **Windows sleep prevention**: `ctypes` call to `SetThreadExecutionState`
  at start — an ~hours-long run would otherwise stall or drop network the
  moment the laptop goes idle.
- **Auto-restart on crash**: `data/run_path2_supervised.sh`, a bash `until`
  loop wrapping the Python script (max 15 restarts) — a network drop or
  unhandled exception at 3am self-heals instead of silently stopping.
- **Resume logic treats error records as "not done"** (not just missing
  files) — a crash-restart or manual rerun retries genuine failures rather
  than permanently skipping them.
- **Periodic auth-token refresh** (every 45 min) — a multi-hour run would
  otherwise start failing with 401s after the ~1hr token expiry.

**Concurrency**: started at 100 workers (validated clean on a 40-doc burst)
before the real quota was found; corrected down to **3 workers** once the
verified Console quota (40 QPM / 10,000 output-TPM) showed only ~1.7 workers
were actually needed — 100 workers were ~90%+ throttled and wasting time on
retries.

**Real result**: **7,130/7,130 documents processed, 65 errors (0.9%,
"unparseable_json" after 6 retries each), 31,621 section pairs**, **414.9
minutes (~6.9 hours)** total → `data/processed/section_pairs_raw.jsonl`.
Close to `PLAN.md`'s ~35,000-pair target (shortfall mostly from legitimately
empty sections, e.g. no clear "judgement" content in shorter headnotes, plus
the 65 failed docs).

---

## 6. Full-corpus HHEM filtering — COMPLETE

**Goal**: apply `score_filter_pairs.py`'s logic (>=80% of a section's claims
must individually score HHEM>=0.5 to be accepted) to all 31,621 real pairs,
producing the real (non-projected) acceptance rate and the final filtered
corpus for M8b fine-tuning.

**First attempt — batching alone didn't help**: rewrote the pilot's per-claim
HHEM scoring into 128-item GPU batches (`data/pilot/hhem_score_batched.py`),
expecting a big speedup on a single L4. Verified via source inspection that
HHEM's own `predict()` already batches correctly internally — the real
problem was a **wrong estimate of total work**: every claim gets checked
against *every* premise chunk of its headnote (not just the relevant one),
so the real corpus has **1,015,479** (chunk, claim) scoring operations, not
the ~212,000 estimated from claim count alone. Measured single-L4 rate: 16.8
batches/min → ~7.8 hours, no real improvement, because the bottleneck was
genuine compute *volume*, not per-call overhead (batching fixes overhead,
not volume).

**Correction to earlier advice given in this session**: "one L4 is enough
for HHEM" was said before this was known, and was wrong once the real
bottleneck turned out to be compute volume — that's exactly the situation
where more GPUs *does* help (unlike the generation step, where it genuinely
didn't matter).

**Current approach — 5-way proportional split** across the user's 4 L4s + 1
A100:
- `data/pilot/shard_corpus.py`: splits `section_pairs_raw.jsonl` into 5
  shards, weighted 1:1:1:1:2.5 (L4:L4:L4:L4:A100 — A100 estimated at 2.5x an
  L4's throughput, later measured closer to **3.1x**).
- `data/pilot/hhem_score_batched.py`: generalized to take a shard-name arg.
- L4 GPU quota is 3/region, so jobs split across regions: 3× L4 in
  `us-central1`, 1× L4 in `asia-southeast1`, 1× A100 in `us-central1`.
- **Real finding**: the corpus has an uneven complexity gradient — later
  documents in the file need more premise chunks per claim. Since the split
  was by raw file position (not shuffled), shard work-item counts came out
  very uneven despite equal pair counts: L4-1 (67,624 items) → L4-2 (89,521)
  → L4-3 (119,847) → L4-4 (219,601) → A100 (518,886, also got more *pairs*).
  **Lesson for next time: shuffle before splitting** to distribute
  complexity evenly across shards.

**Final results, all 5 shards** — this is the real, final, non-projected
number for the whole M1 corpus:

| Shard | Pairs | Accepted | Acceptance | Claims | Mean score | Hallucination |
|---|---|---|---|---|---|---|
| L4-1 | 4,865 | 4,616 | 94.9% | 22,895 | 0.912 | 3.37% |
| L4-2 | 4,865 | 4,698 | 96.6% | 27,771 | 0.933 | 1.85% |
| L4-3 | 4,865 | 4,719 | 97.0% | 30,800 | 0.941 | 1.44% |
| L4-4 | 4,865 | 4,664 | 95.9% | 36,929 | 0.943 | 1.47% |
| A100 | 12,161 | 11,827 | 97.3% | 90,672 | 0.948 | 0.94% |
| **TOTAL** | **31,621** | **30,524** | **96.53%** | **209,067** | **0.9403** | **1.49%** |

(Weighted by claim/pair count, not a simple average across shards — the A100
shard alone accounts for ~38% of the corpus, so it dominates the true mean.)

The full-corpus acceptance rate (96.53%) landed slightly *above* the n=80
pilot's 96.9% projection for Grok 4.1 Fast — essentially confirming the pilot
was representative, not a lucky sample. The A100 shard shows the highest
acceptance/lowest hallucination of any shard, consistent with the earlier
non-shuffled-split finding that later documents in the raw file (routed to
A100, the largest shard) skew toward more, not less, well-behaved content.

A100 hit its OOM (see below) mid-run at 88.8% and resumed cleanly from the
GCS checkpoint after the batch-size/checkpointing fix — no data was lost on
the second attempt.

**Merged corpus-wide outputs** (concatenation of all 5 shards, line counts
verified against the per-shard totals above):
- `data/processed/section_pairs_scored.jsonl` — 31,621 lines (every pair +
  its HHEM claim scores).
- `data/processed/section_pairs_accepted.jsonl` — 30,524 lines (only pairs
  passing the ≥80% claim-pass-rate filter) — this is the corpus M8b actually
  trains on.

**A100 shard failed at 88.8% (batch 3600/4054) — CUDA OOM, ~19GiB single-
batch allocation**, almost certainly a batch that happened to contain
several unusually long premise chunks (attention memory grows with sequence
length, and BATCH_SIZE=128 padding to the longest item in a bad batch could
spike badly). **All progress was lost — the script only uploaded output at
the very end, no incremental checkpointing.** Fixed and resubmitted:
- `BATCH_SIZE` 128 → 32 (keeps worst-case padding memory well within a
  single L4's 24GB, let alone A100's).
- Added a GCS-backed checkpoint every 200 batches (`claim_max_scores` +
  next-batch-index), loaded on startup if present, so a future crash resumes
  instead of restarting from zero.
- Added a per-batch OOM catch that halves and retries the batch recursively
  instead of crashing the whole job outright.
L4-4 finished on the old, unfixed script without ever hitting the OOM — only
A100 was affected. All 5 shards' Vertex Custom Jobs reached
`JOB_STATE_SUCCEEDED`.

**Still to do**:
1. ~~Merge all 5 shards' scored/accepted/summary outputs into corpus-wide
   numbers.~~ **Done** — see merged file paths above.
2. Write the real, final acceptance rate into `data/PROVENANCE.md` per
   `RESEARCH.md` T4 (report plainly, whatever it turns out to be) — not yet
   done.
3. Continue to the rest of the original M1 pipeline: `dedup_split.py`
   (MinHash dedup + 80/10/10 split), `tests/test_data_leakage.py`,
   `pretokenize.py`, `eval/ocr_wer.py` harness.
4. The 200-item human validation pass (separate, human-driven, not
   automatable) to check the HHEM filter's own judgment against a human's.
5. Commit the production generation output and full-corpus HHEM filtering
   results to `m1-data-foundation` and update PR #5 — not yet done.

---

## 7. Files created/modified this session (manifest)

**Documentation**: `RESEARCH.md`, `PLAN.md` (edits, see §1), this file,
`data/rr_summ_alignment_report.md`, `data/PROVENANCE.md` (pending final
write once §6 completes).

**M1 pipeline scripts**: `data/prepare_iltur.py`, `data/align_rr_summ.py`,
`data/build_pairs.py`, `data/score_filter_pairs.py`,
`data/run_path2_production.py`, `data/run_path2_supervised.sh`.

**Pilot/scratch scripts** (`data/pilot/`): `trial_run.py`, `sample_run.py`,
`sample_run_parallel.py`, `sample_run_mistral.py`,
`sample_run_mistral_n80.py`, `sample_run_grok.py`, `sample_run_grok_fast.py`,
`sample_run_grok_concurrency_test.py`, `sample_run_grok_concurrency_test_100.py`,
`hhem_score.py`, `hhem_score_vertex.py`, `hhem_score_batched.py`,
`shard_corpus.py`, `vertex_job_config*.yaml`.

**Config**: `pyproject.toml` (`[data]` extras incl. `google-cloud-aiplatform`),
`.gitignore`, `.claudeignore` (new).

**Data artifacts** (gitignored except where noted): `data/raw/` (IL-TUR
pull), `data/processed/section_pairs_raw.jsonl` (the real production
output), `data/processed/shards/` (sharded HHEM scoring in progress),
`data/pilot/*` result JSONs (small, informative — worth keeping for the
paper's methodology writeup per §4).

**GCP resources created**: project `smartlawai-1` config, billing budget,
GCS bucket `smartlawai-1-m1-pilot`, this session's Vertex Custom Jobs
(all ephemeral, auto-terminate on completion — no standing compute cost).

---

## 8. Dedup, split, pretokenize, OCR-WER harness — COMPLETE (freeze point reached)

All CPU-only, local, no GCP/GPU involved. Full detail and final numbers are
in `data/PROVENANCE.md` §6-8; summarized here.

**`data/dedup_split.py`**: MinHash near-duplicate detection (word 5-grams,
`num_perm=128`, LSH threshold 0.9) over `summ`'s full judgment text, header/
citation-line noise stripped first. Found **29 near-duplicate pairs** (all
exact-content duplicates, jaccard=1.0) across 7,130 documents; dropped 29.
Split the surviving 7,101 docs 80/10/10 with seed 42 →
**train=5,681 / dev=710 / test=710 documents**. Joined the HHEM-accepted
pairs against this frozen split → `data/processed/section_pairs.jsonl`:
**train=24,346 / dev=3,015 / test=3,044 section pairs** (30,405 total; 119
pairs excluded because their source doc was a dropped duplicate — accounts
for all 30,524 accepted pairs exactly). `data/splits/{train,dev,test}.json`
are now frozen per I5.

**Real bug hit and fixed**: the first version of `build_minhash()` called
`MinHash.update()` once per shingle in a Python loop. Across this corpus's
~28.5M total shingles (one judgment alone has 117,512), that ran for 60+
minutes with no end in sight before being killed. Root cause diagnosed via
targeted micro-benchmarks (isolated shingling, minhash-building, and
LSH-insert/query timings on samples) — `.update()`'s fixed per-call overhead
dominates at this scale; `MinHash.update_batch()` (vectorized) does the exact
same computation in under two minutes. **Not a GPU-vs-CPU problem** —
MinHash/LSH is pure hashing/set-membership work with no matrix math, so
cloud GPU compute would not have helped here at all; this was purely an
API-usage bug.

**`tests/test_data_leakage.py`**: 3 tests per `M1_ONBOARDING.md` §9 —
`test_no_doc_id_in_two_splits`, `test_no_near_duplicate_spans_splits` (both
green against the real split/near-dup files), and
`test_no_qa_eval_item_from_training_doc` (correctly skips: no
`eval/gold/*.jsonl` yet, M6 not landed). Full suite: **63 passed, 1 skipped**.

**`data/pretokenize.py`**: InLegalBERT tokenizer (`law-ai/InLegalBERT`,
`max_len=512`), run over the frozen `section_pairs.jsonl` →
`data/processed/section_pairs.arrow/` (30,405 rows, schema: `doc_id, split,
section_id, source_text, target_text, provenance, source_input_ids,
source_attention_mask, target_input_ids, target_attention_mask`). Verified
round-trips via `load_from_disk`. Ran with only `transformers` installed (no
`torch` in this venv) — tokenization alone doesn't need a model backend.

**`eval/ocr_wer.py`**: harness built per `M1_ONBOARDING.md` §11 (reuses
`core/ocr.py`'s `extract_text`, computes WER via `jiwer`). Correctly
hard-fails since `data/ocr_wer_gold/` doesn't exist yet (0/50 gold pages) —
by design, this is human-supplied work that can't be fabricated, and is not
a blocker for the rest of M1. **Decision made this session**: once the 50
hand-transcribed pages exist, the transcripts (small .txt files) will be
**committed** to the repo for reproducibility of the WER number; the page
images stay gitignored (already the case — `.gitignore` only excludes
`data/ocr_wer_gold/pages/`, not `transcripts/`).

`ruff check` clean on all new files.

**M1's literal done-when list is now met**: ~30.4k section pairs exist
(slightly under the ~35k target, explained in `PROVENANCE.md` by the 65
generation errors + near-duplicate drops + HHEM rejections), splits are
frozen and committed with provenance, leakage tests are green, and the
segmentation decision (Path 2) is documented with its rationale. Remaining
open items are explicitly non-blocking: the 200-item human validation pass,
`ACCEPT_THRESHOLD` calibration against it, and the OCR-WER gold set.

---

## 9. OCR-WER gold set: sourced and produced (50/50 real scanned pages)

Built the missing prep tooling and produced the actual 50-page gold set
(pending only human transcription now). All CPU-only, local, no GCP/GPU —
confirmed again this session that neither the download nor detection nor
rendering steps have any compute-scale angle.

**Source hunting hit two real dead ends before working**, each diagnosed
rather than assumed:

1. **Kaggle "SC Judgments India 1950-2024"** (`adarshsingh0903/...`):
   organized as one directory per year, looked ideal. `kagglehub`'s
   per-folder `path=` download 404'd — turned out this specific dataset is
   stored as a single opaque 6.4GB archive blob (`1.archive`), not
   individually-listable files; no per-year targeting is possible against
   it via any tool. Downloaded the full archive instead (succeeded at
   16MB/s after an earlier attempt hit a `ConnectionError` at 26MB — wrapped
   in an auto-retry script, `data/pilot/run_ocr_download_supervised.sh`,
   same pattern as the Path 2 production run's supervisor).
   **Then found the deeper problem**: every 1950 document's text was
   cleanly extractable (0/61 pages flagged as scanned). Inspected the
   extracted text directly — clean structured fields (`Equivalent
   citations:`, `Bench:`, `PETITIONER:`/`RESPONDENT:`) that no 1950s
   typewriter+scan could produce. Root cause: this dataset is *"scraped
   and compiled from Indian Kanoon"*, which hosts judgments as its own
   clean re-transcribed text, not scans of the original filings. No year
   in this dataset would ever contain a genuine scan — a property of the
   source, not fixable by widening the year range.
2. **`vanga/indian-high-court-judgments`** (AWS Open Data, public S3,
   `--no-sign-request`): real per-court/per-year prefixes this time
   (confirmed via direct unauthenticated HTTPS `ListObjectsV2` calls
   against the bucket, no `aws` CLI needed) — `data/tar/year=1950/` etc.,
   each court/bench a separate small tar (whole 1947-1965 range summed to
   ~20MB, checked before downloading anything). **Found a real gotcha the
   docs don't mention**: the S3 `year=` partition is the *case-filing*
   year parsed from the case number, not the decision date — e.g.
   `HCBM020000041950_1_2006-11-21.pdf` sits under `year=1950` but was
   decided in 2006. Filtering by partition alone would have silently
   pulled modern digital documents. Fixed by parsing the real order date
   from each filename's trailing `_YYYY-MM-DD.pdf` and keeping only
   `<= 1970` regardless of which year-partition a file came from
   (`data/pilot/download_ocr_source_pdfs_hc.py`) — 45 genuinely
   old-order-date PDFs kept out of 308 downloaded.
   **Then found a second, subtler problem** in the detector itself: many
   of these 45 PDFs already carry a low-quality OCR text layer embedded by
   whoever digitized them, so `core/ocr.py`'s own `DIGITAL_TEXT_MIN_CHARS`
   text-length heuristic (reused from the serving pipeline) wrongly
   classified them as "digital" — only 2/135 pages flagged. Visually
   confirmed one flagged-"digital" page was genuinely a scan (rendered it,
   looked at it directly: visible scan noise, typewriter irregularities,
   a handwritten margin tick) despite having "clean-looking" extracted
   text. Root cause: the embedded OCR layer produces plausible-length text
   even though the page is an image underneath. Fixed by switching
   detection to check for an embedded raster image covering ≥50% of the
   page area instead of text length (confirmed via `page.images`: the
   genuine scan had a 612×1008 JPEG exactly matching its page's mediabox;
   a genuinely digital "Proceeding Sheet" page had zero embedded images).
   This raised the hit rate to 132/135 pages.

**Result**: 50/50 real scanned pages in `data/ocr_wer_gold/pages/`, all
genuine 1948-1958 Kerala High Court judgments (order dates confirmed via
filename, several visually spot-checked), manifest recorded at
`data/ocr_wer_gold/manifest.json` (doesn't conflict with `.gitignore`'s
`data/ocr_wer_gold/pages/` rule — one directory up). `eval/prepare_ocr_gold.py`
holds the final image-coverage-based detector; the two source download
scripts (`data/pilot/download_ocr_source_pdfs.py` for the abandoned Kaggle
path, `download_ocr_source_pdfs_hc.py` for the working AWS path) are both
kept for the record.

**What's left**: hand-transcribe each of the 50 pages into
`data/ocr_wer_gold/transcripts/{page_id}.txt` (human-only work, per
`RESEARCH.md` T5 — cannot be automated or approximated), then run
`python -m eval.ocr_wer`, which is already built and tested end-to-end
against the hard-fail path.

---

## 10. 200-item human validation pass: sampling + scoring tooling built

Built the tooling for `RESEARCH.md` T4's other required human check —
distinct from §9's OCR-WER work: this validates whether HHEM's automated
accept/reject filter (§5-6) agrees with a human reader, not OCR quality.

**`data/sample_human_validation.py`**: samples 200 items from
`data/processed/section_pairs_scored.jsonl` (the pre-dedup, all-31,621-pairs
file — the only one with both accepted *and* rejected pairs; the frozen
`section_pairs.jsonl` only carries accepted ones). Deliberately **100
accepted + 100 rejected**, not proportional to the corpus's real 96.5%/3.5%
split — proportional sampling would yield only ~7 rejected items, far too
few to say anything about the filter's false-reject behavior. Excludes any
pair whose source document was dropped as a near-duplicate (cross-referenced
against `data/processed/near_duplicate_pairs.json`), so the validation
sample matches documents that actually survived into the real corpus.
Writes blind rater CSVs (`data/human_validation/rater{N}_sample.csv` — no
HHEM verdict shown, to avoid anchoring a rater's judgment) plus a
`answer_key.json` kept separately for scoring. `--num-raters` controls how
many independent copies to generate (2, matching `RESEARCH.md` §7.4's
"silver validation... 2 raters", by default in this session's test run).

**`data/score_human_validation.py`**: takes one or more filled-in rater
CSVs, reports each rater's agreement rate + confusion matrix against HHEM's
verdict (binarizing Accept vs. Revise/Reject — a section a human would
still edit isn't one that should enter training data unedited), and with
2+ raters, pairwise Cohen's κ on the original 3-way verdicts (not the
binarized version, per `RESEARCH.md` §7.5's agreement-reporting mandate).

Verified end-to-end with simulated verdicts (deliberately noisy, ~10-15%
random disagreement, to exercise the agreement math and hard-fail paths —
not real data): confusion matrix and agreement-rate arithmetic checked by
hand, Cohen's κ computed correctly, and both hard-fails (blank verdict,
invalid verdict string) trigger correctly with a clear message. Fresh
**blank** CSVs (`rater1_sample.csv`, `rater2_sample.csv`, 200 rows each,
same seed=42 sample) were regenerated after that test and are what's
actually sitting in `data/human_validation/` now, ready for real raters.
`ruff check` clean.

**What's left**: the actual annotation (two people reading 200
source/generated-text pairs each and marking Accept/Revise/Reject) is
human-only work, not something this session can do. Once both
`rater{N}_sample.csv` files are filled in, run `data/score_human_validation.py`
and write the resulting numbers into `data/PROVENANCE.md` §7, per T4 —
reported plainly regardless of what they turn out to be.

---

## 11. Git / PR status

Branch: `m1-data-foundation`. The original PR
[kramjiy/smartlawai#5](https://github.com/kramjiy/smartlawai/pull/5) was
**merged into `main` on 2026-09-04** (before this session's production-run
and filtering work existed) — pushing new commits to the same branch did
**not** reopen it. A new PR,
[kramjiy/smartlawai#6](https://github.com/kramjiy/smartlawai/pull/6), was
opened for the commits made since that merge (production Path 2 generation +
full-corpus HHEM filtering). The dedup/split/pretokenize/OCR-harness work in
§8 above is not yet committed as of this log entry.
