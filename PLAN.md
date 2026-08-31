# SmartLawAI — Execution Plan

Read `CLAUDE.md` for invariants, `RESEARCH.md` for what we are claiming.
This file is the milestone tracker. One milestone per session.

**Philosophy: skeleton first, then organs.** M0 runs end-to-end with stubs. Every
later milestone swaps one stub for a real component. The system is never broken.

**Compute discipline: M0, M1, M4, M6 need no GPU. M2, M3, M5, M7 use paid GCP L4;
M8 uses L4 (encoders, Arms S/C/M) and A100 (Arm L) — see `CLAUDE.md` §2.** Do not
submit a training job until the eval harness can measure it.

---

## PROGRESS

- [x] **M0** Skeleton — end-to-end with stubs, no GPU
- [ ] **M1** Data foundation — IN-Abs, section pairs, frozen splits
- [ ] **M2** Real retrieval (GCP L4)
- [ ] **M3** Structured generation (GCP L4 endpoint)
- [ ] **M4** ★ Authority registry — the contribution
- [ ] **M5** Verification stack + deterministic gate
- [ ] **M6** Eval harness + gold sets
- [ ] **M7** Baselines (before any training)
- [ ] **M8** Training — encoders, then the QLoRA scale ladder
- [ ] **M9** Analysis, human evaluation, paper

---

## M0 — SKELETON (no GPU, ~2 days)

A request flows through every stage and returns a structured answer. All models stubbed.

**Build:** `config.py` (every threshold) · `scope.py` (frozen dataclass, `I1`) ·
`trace.py` (`PipelineTrace`: stages, retrieval, generation, verification, decision, cost) ·
`pipeline.py` (the single orchestrator) · stub encoder/generator/verifier behind protocols ·
`adapters/local.py` (DuckDB, named-column inserts, `owner_id` on every user table) ·
`api/main.py` (scope required in the request body) · `ui/app.py` (Streamlit).

**Done when:** `pytest -q` green; `cli ingest` then `cli ask --doc-id X` returns a
structured answer with a **complete trace showing every stage**.

> A stage missing from the trace here will be missing forever. Check it.

---

## M1 — DATA FOUNDATION (no GPU, ~4 days) ★ highest leverage

This replaced a week of scraping. Do it properly.

1. **Pull IL-TUR**
   ```python
   load_dataset("Exploration-Lab/IL-TUR", "summ")   # 7,100 judgments + headnotes
   load_dataset("Exploration-Lab/IL-TUR", "rr")     # 21,184 rhetorical-role sentences
   load_dataset("Exploration-Lab/IL-TUR", "lner")   # 105 docs
   load_dataset("Exploration-Lab/IL-TUR", "lsi")    # 66k statute identification
   ```
2. **Section-pair construction — investigate option 1 first (1 day, high payoff).**
   Can `RR` rhetorical roles be aligned to `SUMM` headnotes to yield **gold** section
   pairs? If yes, no silver data anywhere and the methodology is materially stronger.
   If no, fall back to frontier-generated section summaries + 200-item human validation,
   and **record the acceptance rate**.
3. **Dedup then split.** MinHash near-duplicate detection (threshold 0.9) → document-level
   split → `data/splits/{train,dev,test}.json` → `PROVENANCE.md` recording procedure,
   threshold, counts, date. **Freeze. Never regenerate.** (`I5`)
4. **Leakage CI tests** — no `doc_id` in two splits; no near-duplicate spanning splits;
   no QA eval item derived from a training document.
5. **Pre-tokenise** to an Arrow dataset on disk and memory-map it. Faster, reproducible,
   and avoids CPU-side OOM on small-RAM instances.
6. **Measure OCR WER** on 50 hand-transcribed scanned pages → `eval/ocr_wer.md`.
   This number caps everything downstream; know it in week one.

**Done when:** ~35,000 section pairs exist, splits are frozen with provenance, leakage
tests pass in CI, and the segmentation decision is written down with its rationale.

---

## M2 — REAL RETRIEVAL (GCP L4, ~3 days, ~$10–15)

1. **Fix chunking offsets.** Replace `re.split` + `text.find` with a scanning cursor over
   `finditer` positions — the current code returns wrong offsets when clause text repeats,
   which breaks every provenance feature.
2. **Fix parent/child.** Parent = the **full** clause (delete the `[:1200]` truncation).
   Index **children**, return **parents** (small-to-big). Currently both are indexed, so
   near-duplicates crowd the top-k.
3. Add `page_no`, `para_no`, `section_label` to `Chunk` — provenance depends on them.
4. `core/encoders.py` — InLegalBERT mean-pool + L2 norm, `--device` aware.
5. **Scope filter applied inside the vector query**, not after (`I1`).
6. Reranker: `ms-marco-MiniLM` for now — this is the *baseline* reranker; M8 replaces it.

**Done when:** `test_cross_document_isolation` passes and a query over doc A returns
only doc-A chunks, verified in the trace.

---

## M3 — STRUCTURED GENERATION (GCP L4, ~2 days)

Spin up one L4 Vertex endpoint (or a short-lived job) serving Mistral-7B **fp16** —
no Q4 anywhere, so precision never confounds a result.

**Generation returns claims, not prose:**
```json
{"claims":[{"text":"...","passage_ids":["p3"],
            "citations":["S.27 Indian Contract Act, 1872"]}],
 "unanswerable_aspects":["..."]}
```
This is what makes M5's per-claim verification mechanical rather than heuristic.

**Drop HyDE and the LLM judge for now.** Both cost an LLM call; the judge is also
injection-vulnerable. Reintroduce behind flags in M5/M6 only if measurement justifies it.

---

## M4 — ★ AUTHORITY REGISTRY (no GPU, ~5 days) — the contribution

**This is the paper. Protect this week.**

### Schema
```sql
statutes(id, short_title, long_title, year, jurisdiction,
         in_force_from, repealed_on, repealed_by_statute_id)
sections(id, statute_id, number, heading, text,
         in_force_from, in_force_to, superseded_by_section_id)
citation_strings(id, raw, normalised, target_type, target_id, confidence)
```

### Build demand-driven, not exhaustively — you do not need all 511 IPC sections
Legal citations follow a power law.
```python
citations = extract_all_citations(corpus)   # run ner.py over the corpus
top = Counter(citations).most_common()      # build to cover ~95% of citation mass
# typically 100–150 sections, not 511
```
Then **report coverage as a number** — *"covers 94.2% of citations in the corpus;
unresolved citations are flagged, never silently passed."* Stronger than exhaustive,
because it states its own limit (threat T6).

### Parse, then verify a sample — do NOT transcribe by hand
The Ministry of Home Affairs published official IPC↔BNS, CrPC↔BNSS, IEA↔BSA comparison
tables when the new codes commenced. **You are parsing and verifying, not deriving.**
1. Obtain the MHA comparison tables
2. Parse programmatically (~2 h)
3. **Hand-verify a stratified sample of 100 entries** against the Gazette (~1 h)
4. **Report verification accuracy in the paper** — this is also the answer to threat T10

> ⚠️ **Week-1 check: are those tables machine-readable?** Parseable → saves ~20 h.
> Scanned images → costs ~20 h. Largest single unknown in the whole estimate.

### Seed set (from **primary sources** — Gazette of India, eSCR — not secondary summaries)
IPC 1860 → **BNS 2023** · CrPC 1973 → **BNSS 2023** · IEA 1872 → **BSA 2023** ·
Indian Contract Act 1872 · Arbitration & Conciliation Act 1996 ·
Consumer Protection Act 2019 · CPC 1908 · NI Act 1881 · IT Act 2000 · Indian Stamp Act 1899

**The IPC→BNS section-level supersession mapping (effective 2024-07-01) is the single
most important artifact in this project.** Build it carefully; it is what the natural
experiment measures.

### Then
- `registry/resolver.py` — `ner.py` output → normalised citation → registry target
- `verify/registry_check.py` — exists? in force on as-of date? superseded? jurisdiction?
- **Independent audit of 200 entries** by someone who did not build it (threat T10)
- Publish the registry — it is a standalone contribution

**Done when:** `test_repealed_ipc_section_is_struck` passes — a claim citing IPC §420 with
as-of date today is struck and annotated *"repealed 2024-07-01; see BNS §318"*.

---

## M5 — VERIFICATION + GATE (GCP L4, ~4 days, ~$10–15)

1. `verify/structural.py` — every claim carries ≥1 `passage_id`.
2. `verify/entailment.py` — **dual scorer** (`I11`): InLegalNLI (ours, trained in M8a)
   **and** HHEM-2.1, per claim vs its cited passage, both always recorded in the trace.
   **Vendor the HHEM model class into the repo; remove `trust_remote_code=True`** (it
   executes remote code in the serving path). Either failing → `status="unavailable"` (`I2`).
   Until M8a lands, run HHEM-only and leave the InLegalNLI slot returning `unavailable`.
3. **No generative judge in the gate** (`I12`). A classifier cannot be instructed by the
   document it is judging; an LLM can. `verify/sanitize.py` still exists for any advisory
   LLM call outside the gate, but the gate itself sees classifiers only.
4. `gate.py` — deterministic conjunction, fail-closed:
   ```
   structural ∧ entailment ∧ registry        → ANSWER
   any component unavailable                 → REFUSE
   all claims struck                         → REFUSE
   some struck                               → ANSWER remainder + explicit gap notice
   ```
6. **Calibrate the OOS threshold.** The original `RERANK_FLOOR = -10.0` on raw
   cross-encoder logits never fires. Fit on the labelled OOS set; express as a
   probability after Platt scaling.
7. `guardrails/pii.py` — **fix the ordering**: extract entities → store spans separately
   (encrypted) → redact → store. Redacting before extraction destroys NER gold data.
   Entity set: `PHONE`, `EMAIL`, `IN_AADHAAR`, `IN_PAN`, `CREDIT_CARD` only.
   **Never `PERSON`/`LOCATION`/`DATE_TIME`** — they are the substance of a judgment.
   Fail **closed**.
8. **Adversarial suite**: 20 crafted PDFs (instruction override, judge manipulation,
   citation fabrication, exfiltration). No injected instruction may change a gate decision.

---

## M6 — EVAL HARNESS + GOLD SETS (no GPU, ~5 days, mostly human)

**This gates the paper.**

- `eval/metrics.py` — `rouge-score` + `bert-score`. **Delete the hand-rolled ROUGE-L.**
- `eval/stats.py` — paired bootstrap (10k resamples), ECE, reliability diagrams
- **Extractiveness** (4-gram overlap) — mandatory alongside faithfulness
- Gold sets: `retrieval.jsonl` (~200 queries + relevant chunks) · `oos.jsonl` (labelled
  in-scope/OOS/advice-seeking) · `groundedness.jsonl` (claim-level) ·
  **`repealed.jsonl` (criminal-law queries for the IPC→BNS experiment)** ·
  **`entailment.jsonl` (human-labelled (passage, claim, entailed?) — held out, never trained on)**
- **`data/perturbations/`** — synthetic entailment training data generated from IN-Abs
  using the legal perturbation taxonomy (`RESEARCH.md` §2.5). Generator script must be
  releasable. **Disjoint from every evaluation split.**
- Pull **ClaimRAG-Law's 968 validated claims** as the external judge evaluation set

### ★ Build the annotation app first (~4 h) — everything else depends on it
Streamlit, keyboard-driven, one item per screen. It must enforce:
- **Randomised order + blinded system labels** — a rater who knows which output is ours
  is not giving us data
- Correction mode (pre-filled model label) vs cold mode (blank) — we need both, see below
- Logs rater id, timestamp, **time-per-item**
- Live Cohen's κ / Krippendorff's α
- Exports straight to `eval/gold/*.jsonl`

### Annotation techniques (full protocol: `RESEARCH.md` §7)
1. **Model-assisted pre-labelling** — never label cold. 2–3× throughput. Control for
   anchoring with a 10% cold subset.
2. **Pooling** for retrieval relevance (TREC method) — judge the pooled top-10 from
   BM25 / dense / hybrid, ~15 candidates per query.
3. **Duplicate only where subjective**: groundedness, relevance, risk tier = 100%;
   OOS = 30%; **repealed = 0%** (it is an objective registry lookup).
4. **Right-size**: n≈300 → ±5% CI. State the power calculation instead of a round number.

### ⚠️ Pilot 20 items before scaling any set
Compute agreement, fix the ambiguous rubric, **then** run volume. Record items/hour —
that number sizes every remaining set. If it is ugly, cut set sizes **now**, not in week 8.
- `eval/run_eval.py` → results table with mean ± std, written to `eval/results/`
- **CI eval-regression gate** — metrics on a fixed dev subset, blocking on regression

**Done when:** the full table computes for the *untrained* system. That is your floor.

---

## M7 — BASELINES (GCP L4, ~2 days, ~$5) — before any training

They need no training, they set the bar, and they validate the harness on real outputs
for free. If the harness is broken you find out at zero training cost.

| Baseline | Where |
|---|---|
| Mistral-7B fp16, ±RAG | L4 |
| **SaulLM-7B fp16, ±RAG** | L4 |
| **BM25 + frontier model** ← *the one that matters* | API |
| Frontier zero-shot, ±RAG | API |

**Watch for:** if the hybrid+rerank pipeline does not beat BM25 + a good prompt, that is
the most important finding in the project. Surface it now.

**Contamination check (T1):** hold out judgments post-dating model cutoffs; report both.

---

## M8 — TRAINING

### 8a — Encoders (GCP L4, **5 seeds each**, ~$35)
Ordered so each de-risks the next:
1. Document classification (validates the pipeline)
2. NER (feeds the registry resolver; also our L-NER extension)
3. Clause QA span head (replaces the unverified third-party CUAD checkpoint)
4. Risk classifier + **calibration curve, not just F1**
5. Cross-encoder reranker (needs M6 query-passage pairs)
6. ★ **InLegalNLI** — InLegalBERT cross-encoder NLI on `data/perturbations/`.
   ~2 h/run × 5 seeds ≈ $10. Then evaluate: κ vs human labels, head-to-head vs HHEM-2.1,
   **per-perturbation-type detection rates** (the diagnostic that motivates the registry),
   and external generalisation on ClaimRAG-Law.
   **Never train on our own system's outputs.** **Never report it without HHEM alongside** (`I11`).

### 8b — QLoRA scale ladder
**Always:** CPU smoke test (135M model, 5 steps, verify checkpoint **saves and reloads**)
→ dev-only hyperparameter sweep → **freeze config** → seeds.

| Arm | Model | GPU | h/run | Seeds | Cost |
|---|---|---|---|---|---|
| S | Qwen2.5-3B | L4 | ~4 | 3 | ~$10 |
| **C** | **Mistral-7B** (control) | L4 | ~12 | **5** | **~$51** |
| M | Qwen2.5-7B | L4 | ~12 | 5 | ~$51 |
| L | Qwen2.5-14B | A100 | ~7 | 3 | ~$77 |
| | | | | **Total** | **~$190** |

Run seeds **in parallel** — independent runs, same cost, one-third the wall clock.

**If credits run short, cut in this order:** Arm S → Arm L seeds 3→2 → Arm M.
**Never cut Arm C below 5 seeds** — it is the controlled comparison.

**Checkpoint to HF Hub every epoch.** Release the **median** seed and say so.

---

## M9 — ANALYSIS + HUMAN EVAL + PAPER

1. Wire trained components in (reranker, clause head, risk head, doc classifier —
   the last replaces the `doc_type="JUDGMENT"` default that mislabels every contract).
2. **Run `--split test` once.** That is the number.
3. Verification ablation V0→V4 per arm. **V2→V3 is the headline.**
4. **The IPC→BNS natural experiment** (`RESEARCH.md` §2).
5. Retrieval ablations, one variable per condition.
6. Calibration curves; failure taxonomy; error analysis.
7. Human evaluation (4 studies, IAA reported).
8. Regenerate every table from `eval/results/`. **No hand-typed numbers.**

---

## COMPUTE BUDGET

| Item | Cost |
|---|---|
| M2 retrieval dev/test (GCP L4) | ~$10–15 |
| M3 generation endpoint | ~$5 |
| M5 verification+gate dev/test (GCP L4) | ~$10–15 |
| M7 baselines | ~$5 |
| M8a encoders (5 seeds × 6 tasks, incl. InLegalNLI) | ~$35 |
| M8b QLoRA ladder | ~$190 |
| M9 eval generation + demo | ~$10 |
| Frontier API baselines | ~$40 |
| Contingency (reruns — you will need it) | ~$50 |
| **Total** | **~$355–365** |

Now clearly over $300 — moving M2/M5 off the free local 3050 onto paid GCP L4
accounts for most of the increase. Cut Arm S first (~$10), then Arm L to 2 seeds
(~$26). Or accept ~$45–70 out of pocket for the full ladder — it is still the
best-value spend in the project. **Check current GCP L4/A100 on-demand pricing
before treating this table as fixed** — the M2/M5 figures are rough estimates (a
handful of dev/test GPU-hours, not full days billed), not quotes.

---

## PITFALL CHECKLIST — review every milestone

| Pitfall | Guard | Milestone |
|---|---|---|
| Train/test leakage (chunks, duplicate cases, **QA items from LoRA training docs**) | Frozen splits + CI tests | M1 |
| Frontier baselines memorised ILDC | Post-cutoff holdout, both numbers | M7 |
| HHEM rewards verbatim copying | Extractiveness metric | M6 |
| OCR caps the system | WER measured week one | M1 |
| Redaction destroys NER gold | Extract → store spans → redact | M5 |
| Verifier fails open into a fabricated number | `status="unavailable"`, fail closed | M5 |
| Judge steerable by the document it judges | Sanitised context; HHEM primary | M5 |
| Uncalibrated OOS threshold never fires | Platt-scaled, fitted on gold | M5 |
| Weak baselines | BM25 + good prompt is the bar | M7 |
| Ablations changing two variables | One variable per condition | M9 |
| Single-run numbers | 5 seeds, mean ± std, paired bootstrap | M8 |
| **We authored the registry we score against** | Primary sources + independent 200-entry audit + public release | M4 |
| **We trained the judge we score with** | Dual-report with HHEM always (`I11`); κ vs humans; external eval on ClaimRAG-Law; release | M8a |
| Judge trained on data overlapping the eval splits | Perturbations generated from train split only; CI check | M6 |
| Judge learns to like our own model's style | **Never train on our system's outputs** | M8a |
| Synthetic perturbations ≠ natural hallucinations | Validate on human-labelled natural outputs too; report both | M8a |
| Novelty claim stale at submission | arXiv alert now; re-run the search the month you submit | ongoing |
| Lost 20-hour run | Prove reload; push every epoch | M8 |
| Dependency drift | Pin the day it works | M8 |
| Real PII in the demo | Public judgments or synthetic only | M9 |
| **New dataset arrives mid-project → tempting to re-split** | Versioned splits; prefer held-out generalisation set (`OPS.md` §5) | any |
| **Result can't be reproduced later** | `RunManifest` per run, committed (`I5b`) | M8 |
| **Base model silently updated on HF Hub** | Pin `revision=`, not just the name | M8 |
| **Artifact lives only on one laptop / only in GCS** | HF Hub is source of truth; GCS is a cache | ongoing |
| Clean-machine rebuild never tested | Run the §7 recovery test **before** writing the paper | M8 |

---

## FINAL STATE

**System:** upload → OCR → classify → redact → segment → chunk → scoped retrieval →
rerank → structured claims → four-layer verification → deterministic gate → attributed
answer, with every citation resolved against an authority registry and struck if repealed.
Refuses when it cannot verify. Runs on a 3050 + a GCP endpoint.

**Artifacts:** authority registry (public) · **InLegalNLI + its perturbation generator** ·
IndoLexQA · 6 fine-tuned encoders · 4 LoRA adapters (median seed) · full eval tables with
mean ± std and bootstrap CIs · adversarial suite results · reproducible repo with lockfile
and frozen splits.

**Paper:** authority-grounded verification, demonstrated on the IPC→BNS transition.
One sharp contribution executed impeccably beats four asserted ones.

> Ship M0–M7 before you submit a single training job. A measured skeleton beats an
> unmeasured cathedral.
