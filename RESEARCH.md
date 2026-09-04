# SmartLawAI — Research Protocol

What we claim, how we test it, and what would falsify it.
Engineering plan is in `PLAN.md`. Invariants in `CLAUDE.md`.

---

## 1. POSITIONING — what changed and why

The original synopsis claimed four contributions. Two are now occupied by prior work:

| Original claim | Status | Prior art |
|---|---|---|
| "First Indian legal NLP benchmark" | ❌ **Dead** | **IL-TUR** (ACL 2024): 8 tasks, English + Hindi + 9 Indic languages, public leaderboard |
| "First hallucination-aware legal RAG with faithfulness evaluation" | ❌ **Weakened** | **ClaimRAG-Law** (2026): claim-level decomposition + support checking, GDPR + Luxembourg civil law |
| Indian clause taxonomy | ⚠️ Survives, narrow | CUAD is US-only; no validated Indian equivalent |
| Multilingual Indian legal pipeline | ❌ Not supportable as built | Pipeline is OCR-multilingual only |

**Claiming to be first at what IL-TUR already did will get us desk-rejected.**

### The repositioned primary contribution

> **Authority-grounded verification.** ClaimRAG-Law asks *"is this claim supported by
> the retrieved text?"* It does not ask *"does the cited provision exist, and was it in
> force?"* A claim can be perfectly entailed by a retrieved 2019 judgment and still cite
> law repealed in 2024. Entailment-based verification is structurally blind to this.
>
> We resolve every emitted citation against a registry of Indian statutes carrying
> in-force intervals and supersession mappings, and strike or flag claims citing
> provisions not in force on the query's as-of date.

This is unclaimed, safety-critical, legible to a legal audience, and — crucially —
**empirically demonstrable** via the natural experiment below.

### Secondary contributions
2. **InLegalNLI** — a legal-domain faithfulness judge (see §2.5). Its role is to
   *demonstrate empirically* that entailment verification is weakest on statutory
   references — i.e. to prove the gap the registry fills.
3. **IndoLexQA** — Indian legal advisory QA with claim-level citation attribution.
   IL-TUR has **no QA task**; this is a genuine gap, positioned as an *extension* of
   IL-TUR, not a competitor.
4. **Indian clause taxonomy** — 12 Indian categories + ~10 highest-value CUAD types,
   with a calibrated risk classifier (reliability diagram, not just F1).
5. **Scale analysis of legal hallucination** — how hallucination and repealed-citation
   rates vary from 3B to frontier, and whether scale substitutes for verification.

> **Framing discipline:** the registry is the thesis. InLegalNLI is *supporting
> evidence*, not a co-equal headline. A paper with one sharp claim and strong support
> beats one with two headlines competing for space.

### Framed as a general NLP problem, not only a legal one

A 2025–2026 literature check (below) confirms *temporal knowledge grounding* is
an active NAACL/ACL/EMNLP-family subfield right now — *When Facts Change:
Temporal Knowledge Conflict Resolution in LLMs* (ACL 2026 Findings), *DynamicQA*
(EMNLP 2024), *Temporal Validity in Retrieval Memory* (arXiv 2606.26511), *RAG or
Learning? Understanding the Limits of LLM Adaptation under Continuous Knowledge
Drift* (arXiv 2604.05096). None of these use a structured, versioned knowledge
registry, and none touch law. Positioning the authority registry as **a legal-
domain instantiation of temporal knowledge grounding** — rather than only a
legal-AI tool — gives a general-NLP reviewer (NAACL, EMNLP) a body of work to
place this paper against, without displacing the legal framing that already
works for ICAIL/JURIX (§9 Venue). The mechanism doesn't change; only which
literature frames the introduction changes per target venue.

### Extended related-work check (literature scan, August 2026)

| Work | What it does | How this project differs |
|---|---|---|
| **ClaimRAG-Law** (arXiv 2605.21071) | Claim-level RAG benchmark, FR/EN, 968 validated claims; includes "factual recall" questions on citations and effective dates | ⚠️ **Open question, not yet resolved**: the abstract doesn't clarify whether it already checks in-force/currency of a citation vs. just recall-accuracy of a stated date. **Action item**: read the full paper before the "entailment cannot detect repealed law" claim goes into the submitted draft — soften to "differs from" until confirmed |
| **Who Checks the Citations?** (Princeton, arXiv 2606.21155, Aug 2026) | Taxonomy of *fabricated* citation hallucinations from real US court filings, 1,300-item benchmark; finds hallucination rates aren't reliably falling across model generations | Targets fabrication (citing something that doesn't exist), not staleness (citing something real but no longer law) — orthogonal failure mode; cite as motivation, not a competitor |
| **Citation Grounding via Legal Citation Graphs** (arXiv 2606.00898) | Citation-graph-based hallucination detection, Ukrainian jurisdiction | Their own reviewers hit our exact gap and said so in the paper: a citation "we cannot classify — it looks like a repealed provision." Directly supports the motivating claim that citation-*existence* checking alone (their mechanism) still misses repeals — quote this in Introduction |
| **Stanford JELS 2025** (*Hallucination-Free?*) | Preregistered empirical eval of commercial legal AI (Lexis+ AI, Westlaw AI-Assisted Research): 17–33% hallucination rate | Real-world motivating statistic for the introduction; not a method comparison |
| **Falkor-IRAC** (arXiv 2605.14665) | Graph-constrained IRAC-structured legal reasoning generation, Indian judicial AI | Verifies *reasoning structure*, not citation currency; no statute registry |
| **Domain-Partitioned Hybrid RAG for Legal Reasoning ... India** (arXiv 2602.23371) | Neo4j knowledge graph + 3 domain RAG pipelines incl. IPC, LLM-as-judge evaluation | No in-force dates, no IPC→BNS mapping; verification is **LLM-as-judge** — exactly what `I3`/M5 reject as injection-vulnerable. Concrete contrast for related work |

No NLP/ML paper was found mapping IPC↔BNS as a research artifact — only static
lawyer-reference converter sites (ipc2bns.in, legalrath.in). No "InLegalNLI" was
found anywhere. Both confirm the specific angle is still open as of this scan.
**Re-run this search the month of submission** — see T-row in §8 and the
"novelty claim stale at submission" row in `PLAN.md`'s pitfall checklist; this
scan is the dated starting point, not a one-time check.

### Formal definition — authority-grounded (temporal) faithfulness

Standard **entailment-based faithfulness** asks: *is claim C supported by its
cited passage P?* This is necessary but not sufficient when P's own authority
can expire. Define:

> A claim C, citing authority A, is **authority-grounded (temporally faithful)**
> as of date *d* iff (1) C is entailed by its cited passage, **and** (2) A was
> in force on *d* — i.e. A's `in_force_from ≤ d`, A has no `in_force_to` or
> `in_force_to > d`, and A is not `superseded_by` a different authority as of *d*.

Entailment-based faithfulness checks (1) only. This project's contribution is a
mechanism for checking (2) as well, deterministically, against a maintained
registry rather than an LLM's internal (frozen, uncurated) sense of what's
current. The definition is domain-agnostic — anything with versioned or expiring
authorities (statutes, regulations, medical guidelines, safety standards)
qualifies; Indian criminal law's 2024 recodification is the demonstrated
instance, not the boundary of the idea. See `PLAN.md`'s **V2** section for how
a second such instance could be added later without a schema change.

---

## 2. THE CENTREPIECE: the IPC → BNS natural experiment

On **1 July 2024** three foundational Indian criminal statutes were replaced:

| Repealed | Replacement |
|---|---|
| Indian Penal Code, 1860 | Bharatiya Nyaya Sanhita, 2023 |
| Code of Criminal Procedure, 1973 | Bharatiya Nagarik Suraksha Sanhita, 2023 |
| Indian Evidence Act, 1872 | Bharatiya Sakshya Adhiniyam, 2023 |

This gives us something rare: **a dated, unambiguous, high-volume ground truth for
citation staleness.**

Every Indian judgment before that date cites IPC/CrPC/IEA. A RAG system retrieving
pre-2024 case law will faithfully reproduce those citations — grounded, entailed, and
**wrong for a query about present-day law**.

### The measurement
For a set of criminal-law queries with as-of date = today:

| Condition | Predicted |
|---|---|
| Retrieved passages citing repealed provisions | High (most case law pre-dates July 2024) |
| Claims entailed by their cited passage | High — the system is behaving correctly |
| **Claims citing provisions not in force** | **The number nobody reports** |
| Caught by entailment verification | **~0%** — entailment is blind to this |
| Caught by authority registry | Measurable, and the contribution |

### Why this is a strong result
- It is a **failure mode of the current state of the art**, not a strawman.
- It is **reproducible** — the dates are public law.
- It has **direct citizen harm potential**, which matters for a legal-AI venue.
- It generalises: every jurisdiction has repeals; India just gave us a clean one.

**If this experiment produces a low repealed-citation rate, say so.** A negative result
here is still publishable ("modern models largely avoid repealed provisions, but N% of
grounded answers still cite them") and it is honest.

---

## 2.5 InLegalNLI — the faithfulness judge

### Why train one at all
HHEM-2.1 is general-domain and requires `trust_remote_code=True` — arbitrary remote code
in the serving path. A domain-trained classifier is faster, safer, and — critically —
**injection-immune**. A classifier cannot be instructed by the document it is judging;
a generative judge can. This closes the sharpest hole in the original architecture,
where the LLM judge received the same unsanitised context it was meant to police.

### Architecture
**InLegalBERT fine-tuned as a cross-encoder NLI model**: `(passage, claim) → entailed / not`.
~110M params, runs on the 3050, milliseconds per claim, no `trust_remote_code`.
It also reuses the thesis encoder — retrieval, reranking, and verification all InLegalBERT.

**Not a generative judge** (`I12`). Ever, inside the gate.

### Training data — synthetic perturbation of IN-Abs
The gold groundedness set is *evaluation*, never training. Generate training pairs by
taking claims genuinely entailed by a passage and corrupting them in controlled ways.
Design perturbations around **legal** failure modes, not generic NLI:

| Perturbation | Legal failure mode |
|---|---|
| Change section number | Wrong statutory provision |
| **Substitute IPC↔BNS incorrectly** | **Repealed / superseded law** |
| Negate obligation (`shall` → `shall not`) | Reversed legal effect |
| Swap appellant / respondent | Misattributed holding |
| Alter date or monetary amount | Factual error |
| Change court name | Wrong precedential weight |
| Insert unsupported holding | Fabricated ratio |

### The result that ties this to the registry
Report **detection rate per perturbation type.** Predicted finding: entailment models
catch negations and party swaps easily but perform poorly on **section-number
substitutions** — "Section 420" and "Section 421" are near-identical in embedding space
and the surrounding text is unchanged.

If that holds:

> *"Faithfulness models detect ~94% of negation errors but only ~61% of statutory-reference
> errors. Citation correctness is precisely where entailment-based verification is weakest —
> and precisely what authority-level verification addresses."*

That **demonstrates** the gap the registry fills rather than asserting it. This is the
reason to build the judge.

### Evaluation of the judge itself
1. **Agreement with human annotators** on a held-out set → Cohen's κ. An unvalidated
   judge is a liability; a κ≈0.8 judge is an instrument.
2. **Head-to-head with HHEM-2.1** on the same items.
3. **ClaimRAG-Law's 968 human-validated claims** — external, independent, out-of-domain
   (GDPR / Luxembourg civil law). Tests generalisation *and* removes the "you built your
   own test set" objection for at least one result.
4. **Per-perturbation-type breakdown** on the synthetic set.

### Hard rules
- **Dual-report always** (`I11`). InLegalNLI supplements HHEM, never replaces it.
- **Never train on our own system's outputs** — the judge would learn to like our style.
- **Judge training data disjoint from every evaluation split.**
- **Release the model and the perturbation generator.**

---

## 3. DATA

### Use, do not rebuild

| Source | Size | Use |
|---|---|---|
| **IL-TUR `SUMM`** (= IN-Abs) | **7,100 SC judgments + professional headnotes** | Summarisation gold. **Replaces scraping.** Leaderboard-comparable |
| **IL-TUR `RR`** | 21,184 sentences, rhetorical-role labels | **Section segmentation for free** — check whether this yields section-aligned pairs with no silver generation |
| **IL-TUR `LSI`** | 66k docs, statute identification | Statute-linking evaluation; registry stress test |
| **IL-TUR `L-NER`** | **105 docs** | Legal NER. Tiny — a serious extension is itself a contribution |
| **IL-TUR `CJPE`** (= ILDC) | 34,816 SC judgments | Retrieval corpus; contamination analysis |
| **IL-TUR `BAIL`** (= HLDC) | ~176k Hindi district docs | Only if the multilingual arm is revived |
| **CUAD** | 510 US contracts, 41 clauses | Clause baseline; source of the ~10 imported types |
| **Indian contracts** | to collect | The 12 Indian clause types. Real annotation needed |

### Section-pair construction (decide in M1, record the decision)
1. **Preferred:** align `RR` rhetorical roles to `SUMM` headnotes → gold section pairs, no silver.
2. **Fallback:** generate section summaries with a frontier model, human-validate a
   random 200, **report the validation rate in the paper.** Silver data is normal;
   concealing it is not.

Expected: 7,100 documents × ~5 sections ≈ **35,000 section pairs**, which honestly
supports a 14B arm.

### Splits (`I5`)
Near-duplicate dedup (MinHash over normalised text) → **document-level** split →
frozen JSON → `PROVENANCE.md` recording the procedure, the dedup threshold, and counts.
Never regenerated.

**Leakage checks that must run in CI:**
- No `doc_id` in two splits
- No QA eval item derived from a document in the LoRA training set
- No near-duplicate pair spanning splits (cosine > 0.95)

---

## 4. EXPERIMENTAL DESIGN

### 4.1 Model arms

| Arm | Model | Fine-tuned | Purpose |
|---|---|---|---|
| S | Qwen2.5-3B-Instruct | LoRA | Lower bound; hallucinates most |
| **C** | **Mistral-7B-Instruct-v0.3** | LoRA | **Control** — same base as SaulLM-7B |
| M | Qwen2.5-7B-Instruct | LoRA | Method transfers across bases (Apache-2.0) |
| L | Qwen2.5-14B-Instruct | LoRA | Does scale substitute for verification? |
| ∞ | GPT-5.x / Claude / Gemini | ✗ zero-shot | Frontier ceiling, via API |

**Arm C is load-bearing.** SaulLM-7B = Mistral-7B + legal continued-pretraining;
ours = Mistral-7B + legal LoRA. Same base, two adaptation strategies — differences are
attributable to method, not foundation.

### 4.2 Baselines (never fine-tuned — that is what makes them baselines)
- Mistral-7B zero-shot, no RAG
- Mistral-7B + our RAG
- **SaulLM-7B** zero-shot, and + RAG
- **BM25 + frontier model** ← *the baseline that matters*. If our hybrid+rerank pipeline
  does not beat a good prompt over BM25, that is the finding. Learn it in M5, not month six.
- Frontier zero-shot, and + RAG

### 4.3 Verification conditions (the ablation that carries the paper)

| # | Condition | Isolates |
|---|---|---|
| V0 | No verification | Raw hallucination + repealed rates |
| V1 | Structural only (every claim carries a passage id) | Attribution alone |
| V2 | V1 + entailment (HHEM) | **State of the art** |
| V3 | **V2 + authority registry** | **Our contribution** |
| V4 | V3 + LLM judge (sanitised) | Marginal value of the judge |

**V2 → V3 is the headline delta.** Report it per model arm.

### 4.4 Retrieval ablations — one variable per condition
BM25-only · dense-only · RRF · RRF+rerank · ±HyDE · ±clause-boundary chunking ·
parent-return vs child-return

> If a condition changes two things, it measures nothing. This is the single most
> common ablation error.

---

## 5. METRICS

| Family | Metric | Tool |
|---|---|---|
| Retrieval | Recall@{5,10,20}, MRR, nDCG@10 | in-house, tested |
| Summarisation | ROUGE-1/2/L | **`rouge-score`** — never hand-rolled |
| | BERTScore | `bert-score` |
| | Faithfulness | HHEM-2.1, claim-level |
| | **Extractiveness** | 4-gram overlap with source |
| QA | Claim-level groundedness | **InLegalNLI + HHEM-2.1, both reported** (`I11`) |
| | **Citation validity rate** | registry resolution |
| | **Repealed-citation rate** | registry + in-force interval |
| Judge | Agreement with human labels | Cohen's κ, held-out set |
| | Head-to-head vs HHEM-2.1 | accuracy / F1, same items |
| | **Detection rate per perturbation type** | synthetic set — the diagnostic |
| | External generalisation | ClaimRAG-Law, 968 validated claims |
| Refusal | Precision / recall on labelled OOS | gold OOS set |
| Calibration | **ECE + reliability diagram** | `eval/stats.py` |
| Cost/latency | tokens, USD, p50/p95 per stage | `PipelineTrace` |

### Why extractiveness is mandatory
HHEM rewards entailment. A model that copies source sentences verbatim scores near-perfect
faithfulness and is a useless summariser. **Reporting faithfulness without extractiveness
invites the reviewer question "did you just train a copier?" and we must have the answer.**

### Why calibration is mandatory
We display a faithfulness score to non-expert users making consequential decisions.
Does 0.8 mean 80% correct? A reliability diagram is cheap, under-reported in legal NLP,
and a genuine reviewer-pleaser.

---

## 6. STATISTICAL PROTOCOL

- **5 seeds** (`42, 1337, 2024, 31337, 8191`), pre-registered, mean ± std always.
- Hyperparameters tuned on **dev only**, then frozen; seeds run after. Selecting a seed
  by test score is test-set overfitting.
- **Paired bootstrap** over test items (10,000 resamples) for system comparisons —
  not a t-test over 5 seeds, which is underpowered.
- **Bonferroni correction** where multiple systems are compared on one metric.
- Report **effect size**, not just significance. A significant 0.3 ROUGE point is noise.
- Baselines get the same seed treatment as our system. A 5-seed mean against a
  single lucky baseline run is not a comparison.

---

## 7. ANNOTATION PROTOCOL

**Nothing here is labelled one-by-one from scratch.** Four techniques do most of the work.

### 7.1 The four techniques

**1. Model-assisted pre-labelling — never label cold.**
A frontier model proposes a label; the human accepts or corrects. **2–3× throughput.**
Standard practice, but it must be reported, and **anchoring must be controlled**: have one
rater label a 10% subset *cold* and compare against the assisted labels. If assisted
labels agree with the model far more than the cold subset does, you have anchoring —
report it and discount accordingly.

**2. Pooling for retrieval relevance** (TREC methodology).
Judging every chunk against every query is impossible. Run BM25, dense, and hybrid;
pool the top-10 from each; judge only the ~15 unique candidates per query. Established,
citable, and it reduces 200 queries from millions of judgments to ~3,000.

**3. Duplicate only where subjectivity lives.**

| Gold set | Duplicate for IAA? | Rationale |
|---|---|---|
| Groundedness | ✅ 100% | Subjective |
| Retrieval relevance | ✅ 100% | Subjective |
| Risk tier | ✅ 100% | Most contested of all |
| OOS classification | ⚠️ 30% | Largely objective |
| **Repealed / not** | ❌ 0% | It is a registry lookup — objective by construction |

**4. Right-size with a power calculation, not a round number.**
For a proportion, n≈300 gives ≈±5% CI; n≈500 gives ≈±4%. Diminishing returns beyond.
**300–400 per set, with the calculation stated**, is more defensible than 1,000 unjustified.

### 7.2 Build the annotation app (≈4h, `PLAN.md` M6)

Better than a spreadsheet, and better than off-the-shelf tools for our rubrics because it
enforces the things that protect validity:

- One item per screen, keyboard-driven
- **Randomised order and blinded system labels** — a rater who knows which output is ours
  is not giving us data
- Optional pre-filled model label (correction mode) vs blank (cold mode)
- Logs rater id, timestamp, **time-per-item**
- Computes Cohen's κ / Krippendorff's α live
- Exports directly to `eval/gold/*.jsonl`

### 7.3 Always pilot 20 items first
Compute agreement, find where the rubric is ambiguous, rewrite it, **then** scale.
A rubric run at full volume before piloting produces an unusable α and wastes everyone's
time. Record the pilot's items/hour — it is how you size everything else.

### 7.4 The studies

| Study | N | Raters | Measures |
|---|---|---|---|
| Answer quality | 200 | 3 law students + **2 practising advocates** | 5-point: accuracy, completeness, clarity |
| **Citation audit** | 200 citations | 2 | Exists? In force? Correctly applied? |
| Summary usefulness | 100 | 3 | 5-point + "would you rely on this" |
| Silver validation (if Path 2 in §3) | 200 | 2 | Accept / revise / reject |

**Recruit realistically.** Advocates are scarce, students are not. Use students for volume
and 1–2 advocates for a smaller expert subset, then **report results stratified by rater
type** — that comparison is itself an interesting result.

**Compensate raters**, even modestly. Volunteer annotation has severe attrition, and
losing a rater mid-study costs you the agreement statistic entirely.

### 7.5 Report agreement for every study
Cohen's κ (2 raters) / Krippendorff's α (3+). **Expect risk-tier agreement to be low** —
legal risk is genuinely contested among practitioners. A low α there is a finding about
the task, not a failure of the annotators. Report it and discuss it.

---

## 8. THREATS TO VALIDITY — address each explicitly in the paper

| # | Threat | Mitigation | Where |
|---|---|---|---|
| T1 | **Contamination** — ILDC (2021) and Indian Kanoon are public and pre-date every frontier model | Hold out judgments post-dating model cutoffs where establishable; report both numbers; discuss the gap | §Results |
| T2 | **Leakage** — QA eval items derived from LoRA training documents | Document-level frozen splits + CI check | §Data |
| T3 | **Metric gaming** — faithfulness via verbatim copying | Extractiveness reported alongside | §Metrics |
| T4 | **Silver data** — model-generated section summaries | Human-validate 200, report acceptance rate | §Data |
| T5 | **OCR ceiling** — 80% WER caps everything downstream | Measure WER on 50 hand-transcribed pages; report it | §Limitations |
| T6 | **Registry coverage** — we seed ~10 statutes, not all Indian law | Report coverage rate; out-of-registry citations flagged, not silently passed | §Limitations |
| T7 | **Single jurisdiction** | Frame as a method demonstrated on India, generalisable to any jurisdiction with repeals | §Discussion |
| T8 | **Prompt-injection via documents** | Adversarial suite of 20 crafted PDFs; report gate integrity | §Security |
| T9 | **Annotator expertise** — students ≠ advocates | Stratify results by rater type; report both | §Human eval |
| T10 | **Author-built registry** — we define the ground truth we score against | Registry built from primary sources (Gazette, eSCR); 200-entry independent audit; released publicly | §Data |
| T11 | **Author-built judge** — we trained the model that scores our faithfulness | **Dual-report with HHEM-2.1 always** (`I11`); human-agreement κ reported; external evaluation on ClaimRAG-Law; judge released | §Judge |
| T12 | **Synthetic training data** for the judge may not reflect natural hallucinations | Validate on *human-labelled natural* system outputs, not only perturbations; report both | §Judge |

**T10 and T11 are the same structural weakness**, and together they are the sharpest
objection available to a reviewer: we authored both the registry and the judge, then
scored ourselves with them. Pre-empt this explicitly and early in the paper.

The answer is the same in both cases — **primary sources, independent audit, external
validation, dual reporting, public release.** If our conclusions hold under HHEM-2.1
as well as InLegalNLI, and the registry survives an audit we did not perform, the
objection is answered with evidence rather than assurance.

---

## 9. PAPER OUTLINE

1. Introduction — the grounded-but-repealed failure mode
2. Related Work — IL-TUR, ClaimRAG-Law, SaulLM, CUAD, HHEM/Self-RAG, temporal-knowledge-
   conflict literature (*When Facts Change*, *DynamicQA*), *Who Checks the Citations?*,
   *Citation Grounding via Legal Citation Graphs*, Falkor-IRAC, Domain-Partitioned Hybrid
   RAG (§1 extended related-work table). **Position as extension, not competitor**
3. The Authority Registry — schema, in-force intervals, supersession, sourcing
4. **InLegalNLI** — architecture, perturbation taxonomy, per-type detection.
   **Place this immediately before Results** so the "entailment is weak on statutory
   references" finding lands right before the registry results that exploit it
5. System — pipeline, scoped retrieval, structured claims, layered verification, gate
6. IndoLexQA — construction, annotation, agreement
6. Experimental Setup — arms, baselines, ablations, splits, seeds
7. Results — V2→V3 delta per arm; scale ladder; retrieval ablations
8. **The IPC→BNS Case Study** — the natural experiment
9. Human Evaluation — quality, citation audit, agreement
10. Calibration and Error Analysis — failure taxonomy
11. Threats to Validity
12. Limitations and Ethics — BCI disclaimer, DPDPA, bias audit, no-legal-advice boundary
13. Conclusion

### Venue
**ICAIL or JURIX preferred** — legal-AI reviewers will engage with in-force dates,
supersession, and court hierarchy, and will recognise why IPC→BNS matters.
ACL/EMNLP reviewers will ask about benchmarks and baselines and may undervalue the
legal-correctness work that is our strongest asset.
Weight the paper toward the registry for a legal venue; toward IndoLexQA and the
ablations for an NLP venue.

**For a NAACL/EMNLP/ACL-family submission specifically**: lead Introduction/
Related Work with the temporal-knowledge-grounding framing and general-domain
literature from §1's "Framed as a general NLP problem" subsection, not only
legal-AI framing — that's what gives this audience a body of work to place the
paper against. The registry stays the headline result either way; only the
framing of *why it matters* shifts per venue.

---

## 10. REVIEWER OBJECTIONS — have these answers ready

| Objection | Answer |
|---|---|
| "IL-TUR already benchmarks Indian legal NLP" | We extend it. IL-TUR has no QA and no clause extraction; our numbers are leaderboard-comparable on shared tasks |
| "ClaimRAG-Law already does claim-level verification" | Entailment-based checking differs from authority/in-force checking (§1 flags an open question here — confirm against ClaimRAG-Law's full text before final draft, not just its abstract). We show V2 catches ~0% of repealed citations; V3 catches most |
| "Doesn't citation-graph grounding (e.g. arXiv 2606.00898) already solve this?" | No — that paper's own review found a citation "we cannot classify — it looks like a repealed provision." Citation graphs check existence, not temporal validity |
| "Isn't this the same as fabricated-citation detection (e.g. Who Checks the Citations?, arXiv 2606.21155)?" | Different failure mode: they detect citations to things that don't exist; we detect citations to things that exist but are no longer law |
| "Why not a bigger model?" | Scale ladder to 14B; dataset size (35k pairs) caps honest capacity; frontier models included as the ceiling |
| "You built the registry you score against" | Primary sources, independent 200-entry audit, publicly released |
| "You built the judge you score with" | Dual-reported with HHEM-2.1 throughout; κ vs human labels; externally evaluated on ClaimRAG-Law; released |
| "Synthetic perturbations aren't real hallucinations" | Judge validated on human-labelled natural system outputs as well as synthetic; both reported |
| "Does this generalise beyond India?" | The mechanism is jurisdiction-agnostic; India provides an unusually clean dated repeal |
| "Is a 7B good enough for legal advice?" | We make no such claim. The system provides *information* with enforced attribution and refuses when it cannot verify |
| "Contamination?" | §Results, post-cutoff holdout, both numbers reported |

---

## 11. WHAT WOULD FALSIFY THE THESIS

State these before running anything — it is what makes it science.

- Repealed-citation rate is near zero even at V0 → the problem does not exist at scale;
  report it and pivot the paper to the negative result
- V2 already catches repealed citations → entailment is not blind after all; the
  contribution collapses; report why
- Registry coverage is so low that most citations are "unknown" → mechanism impractical
  at realistic coverage; report the coverage/utility curve
- **InLegalNLI detects statutory-reference errors as well as it detects negations** →
  entailment is not blind to citation errors after all; the motivating gap narrows and
  the registry's value must be argued on in-force/supersession alone (which still stands)
- **InLegalNLI fails to beat HHEM-2.1** on human-labelled data → drop it from the metric
  stack, report HHEM only, and keep it as a negative result about domain adaptation

Pre-registering these is a strength. Discovering them at review time is not.

---

## 12. V2 — DEFERRED EXTENSIONS (research rationale)

Full engineering detail (what's cheap later because of a V1 design choice made
now) lives in `PLAN.md`'s own `## V2` section. The research case for each,
briefly:

- **A second independent natural experiment** (another Indian repeal event
  beyond IPC/CrPC/IEA) would turn "one dated example" into a small benchmark of
  temporal-validity events, meaningfully strengthening the generalisation claim
  in T7/§10 ("does this generalise beyond India?") beyond assertion.
- **Publishing the repealed-citation set as a standalone diagnostic benchmark**
  (not just an internal eval slice) is what turns "we measured this" into "we
  defined a task" — a citable resource contribution independent of this
  system's own score on it.
- **Expert (practising-lawyer) annotation for IndoLexQA**, replacing or
  supplementing crowd/student annotation, directly strengthens T9 (annotator
  expertise) and is exactly the kind of costly-to-produce gold data that legal-
  AI venues weight heavily.
- **Multilingual verification** (Hindi/Indic judgments, not just OCR) would
  make the "multilingual pipeline" claim killed in §1's positioning table
  actually supportable — but honestly requires new encoder validation and eval
  data, closer to a second project than an extension. Not assumed anywhere in
  V1's design; tracked here so it isn't silently forgotten either.

None of these are committed scope. They're documented so that, if pursued
later, they extend cleanly rather than requiring rework of V1 decisions.
