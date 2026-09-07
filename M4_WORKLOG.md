# M4 — Authority Registry: Work Log

**Project:** SmartLawAI — hallucination-aware legal RAG for Indian law
**Milestone:** M4, the authority registry (`PLAN.md` M4 — *"This is the paper. Protect this week."*)
**Date:** 5 September 2026
**Status:** M4's stated done-when condition is **met**. Exhaustive ingestion substantially complete; verification outstanding.

---

## 1. What M4 is, in one paragraph

A database of Indian statutes and their sections, each carrying an in-force
interval and a supersession mapping, plus a resolver that answers one question:
*"this citation, as of this date — is it still law?"* It exists because
entailment-based verification is structurally blind to staleness. A model can
retrieve a 2019 judgment, summarise it faithfully, cite IPC §420 — perfectly
grounded, and wrong law, because the IPC was repealed on 2024-07-01. `RESEARCH.md`
§1 makes this the paper's repositioned primary contribution.

**Done when:** `test_repealed_ipc_section_is_struck` passes — a claim citing
IPC §420 as of today is struck and annotated *"repealed 2024-07-01; see BNS §318"*.
**This now passes.**

---

## 2. Paths

### Working root (current)
```
C:\Users\ramch\smartlaw\M4\smartlawai-1
```
All M4 work lives in the `M4/` subfolder of that repo, self-contained, touching
nothing under `src/smartlawai/`.

### Previous root (superseded)
```
C:\Users\ramch\OneDrive\Desktop\M4\smartlawai-1
```
Used only at the very start, to read the four project docs. **No M4 work was
ever written there** — that copy contains no `M4/` folder. The one change made
there was deleting a broken `.venv` (its `pyvenv.cfg` pointed at
`C:\Users\kasir\AppData\Local\...`, a path belonging to a different Windows user
on a different machine, so every command through it failed). That deletion
happened *before* the path switch and was disclosed at the time. The folder has
since been restored and kept.

The two copies have diverged (Desktop venv: 293 package dirs; smartlaw venv: 177).
Both function. Only `smartlaw` is current.

### Environment
- Python **3.13.15** at `C:\Users\ramch\AppData\Local\Programs\Python\Python313\python.exe`
- venv at `<root>\.venv` — **invoke by absolute path**; bare `python` on PATH is a
  Microsoft Store stub that does nothing. There is no `py` launcher.
- Project targets 3.11; 3.13 works. Installed with `pip install -e ".[dev,data]"`.
- Key versions: `duckdb 1.5.5`, `datasets 2.21.0`, `pyarrow 25.0.1`, `pytest 9.1.1`.
- **spaCy and torch are NOT needed.** `core/ner.py` falls back to regex, and
  `STATUTE`/`PROVISION` — the only labels M4 consumes — come from the regex path.
  This avoided a multi-GB install.

---

## 3. Complete file inventory

All paths relative to `C:\Users\ramch\smartlaw\M4\smartlawai-1\`.

### Library code — `M4/registry/` and `M4/verify/` (700 lines)

| File | Lines | Purpose |
|---|---|---|
| `M4/registry/types.py` | 48 | **Frozen M4↔M5 contract.** `ResolvedCitation`, `RegistryResult`, four status constants |
| `M4/registry/normalise.py` | 186 | Act-alias → canonical id, section-ref parsing, `is_compatible()`, abbreviations |
| `M4/registry/schema.py` | 151 | DuckDB schema: `STATUTES`, `SECTIONS`, `SUPERSESSION`, `CITATION_STRINGS` |
| `M4/registry/resolver.py` | 88 | `resolve_citation(raw) -> ResolvedCitation \| None` |
| `M4/verify/registry_check.py` | 145 | `check_citation(cit, as_of) -> RegistryResult`. Never raises |
| `M4/registry/ingest/mha_tables.py` | 213 | Parses the NCRB correspondence tables into supersession rows |
| `M4/registry/ingest/indiacode.py` | 82 | India Code DSpace REST client — section lists per act |
| `M4/conftest.py` | 8 | Puts `M4/` on `sys.path` for pytest |

### Tests — `M4/tests/`

| File | Lines | Contents |
|---|---|---|
| `M4/tests/test_registry.py` | 320 | 24 tests. Includes the done-when test, repeal-boundary, sedition, I2 guarantees, normalisation invariants |

### Scripts — `M4/scripts/` (1,441 lines)

**Pipeline (run in this order to rebuild everything):**

| # | Script | What it does |
|---|---|---|
| 1 | `seed_statutes.py` | Seeds 14 statutes with commencement/repeal dates |
| 2 | `parse_mha.py` | Parses correspondence tables; ground-truth spot checks |
| 3 | `load_registry.py` | Loads mappings into `SECTIONS` + `SUPERSESSION` |
| 4 | `find_act_ids.py` | Resolves India Code `act_id`s → `M4/out/act_ids.json` |
| 5 | `ingest_indiacode.py` | Pulls all sections + headings for 10 statutes |
| 6 | `validate_sections.py` | Cross-validates parsed vs authoritative section lists |
| 7 | `extract_citations.py` | Citation frequency analysis over the 7,130-judgment corpus |
| 8 | `coverage_report.py` | Coverage curve, per-act breakdown, orphan analysis |
| 9 | `backfill_headings.py` | Recovers IPC/CrPC/IEA headings + missing sections from the old-code column |
| 10 | `ingest_constitution.py` | Parses and loads 466 Constitution articles |
| 11 | `resolution_rate.py` | Measures registry resolution + staleness over the corpus |
| 12 | `load_citation_strings.py` | Populates CITATION_STRINGS from the corpus, resolved and unresolved |
| 13 | `make_audit_sample.py` | Draws the stratified 100/200 verification samples |
| 14 | `score_audit.py` | Scores filled worksheets: accuracy + Cohen's kappa |
| — | `demo.py` | End-to-end demo: citation → target → verdict |
| — | `md_to_pdf.py` | Renders this work log to PDF |

**Diagnostics (one-off, kept for reproducibility):**
`triage_pdf.py` (scanned-vs-text verdict), `inspect_ncrb.py`, `inspect_corpus.py`,
`inspect_orphans.py`, `check_unparsed.py`, `probe_bare_act.py`.

### Source data — `M4/registry/data/` (7.7 MB)

| File | Source | Notes |
|---|---|---|
| `mha/ncrb_BNS.html` | cytrain.ncrb.gov.in | 600 rows — **primary parse target** |
| `mha/ncrb_BNSS.html` | cytrain.ncrb.gov.in | 636 rows |
| `mha/ncrb_BSA.html` | cytrain.ncrb.gov.in | 209 rows |
| `mha/BNS_IPC_Comparative_uppolice.pdf` | uppolice.gov.in | 23 pp — corroboration |
| `mha/BNSS_CrPC_comparison_bprd.pdf` | bprd.nic.in | 38 pp — corroboration |
| `mha/BSA_IEA_comparison_bprd.pdf` | bprd.nic.in | 14 pp — corroboration |
| `mha/MHA_BNS_2023_bare_act.pdf` | mha.gov.in | 407,775 chars |
| `mha/MHA_BNSS_2023_bare_act.pdf` | mha.gov.in | 857,117 chars |
| `mha/MHA_BSA_2023_bare_act.pdf` | mha.gov.in | 169,980 chars |
| `mha/PROVENANCE.md` | — | Full sourcing log + parse notes |
| `indiacode/PROVENANCE.md` | — | API docs, ingestion results, known gaps |
| `constitution/COI_english_2025-11.pdf` | legislative.gov.in | 799 pp, 1.94M chars — English-Kannada diglot |
| `constitution/PROVENANCE.md` | — | Edition caveat, omitted-article list, honesty note |

### Generated outputs — `M4/out/` (5.9 MB)

| File | Contents |
|---|---|
| `registry.duckdb` | **The registry.** 14 statutes, 3,817 sections, 1,289 supersessions |
| `citations.parquet` | 75,184 citation rows extracted from the corpus |
| `citation_frequency.json` | Frequency tables, coverage curve, attribution breakdown |
| `act_ids.json` | India Code `act_id` for 10 statutes (committed so lookup runs once) |
| `indiacode_ingest_report.json` | Per-act ingestion counts |
| `section_validation.json` | Cross-validation results |
| `extract_full.log` | Full corpus run log |
| `ic_*.json`, `probe_bns.txt` | **Scratch** from API discovery — safe to delete |

---

## 4. What was done, in order

### Step 1 — Environment
Found the venv non-functional (pointed at another user's Python). Rebuilt on
3.13, installed `[dev,data]`. Baseline suite: 49 passed, 1 failed, 11 errors.
Characterised both failures as **unrelated to M4**: the 11 errors are missing
`fastapi` (we installed `[dev,data]`, not `[api]`); `test_faiss_lifecycle` is
**flaky, not broken** — it passed 3/3 on rerun because it seeds
`np.random.rand` with nothing. `tests/test_ner.py` passes 16/16.

### Step 2 — Workspace and frozen contract
Created `M4/` and wrote `types.py` to the exact shape in `M4_ONBOARDING.md` §6,
so M5's `gate.py` integration is a non-event.

### Step 3 — The week-1 unknown, resolved
`PLAN.md` calls this *"the largest single unknown in the whole estimate"*:
parseable tables → ~2h, scanned → ~20h.

**All tables are machine-readable.** Verdicts from `triage_pdf.py`:

| Document | Chars | Images | Verdict |
|---|---|---|---|
| BNS↔IPC (UP Police) | 66,591 | 0 | MACHINE-READABLE |
| BNSS↔CrPC (BPRD) | 75,644 | 38 (page logos) | MACHINE-READABLE |
| BSA↔IEA (BPRD) | 28,511 | 14 (page logos) | MACHINE-READABLE |

The ~20h OCR branch is closed. No Tesseract or poppler needed.

### Step 4 — Sourcing, with corroboration
Every mapping has **two independent official sources**. NCRB (National Crime
Records Bureau) and BPRD (Bureau of Police Research & Development) are both
**Ministry of Home Affairs bodies** on official government domains — materially
better provenance than the state-police PDF found first.

**Correction to the project docs:** `PLAN.md` and `M4_ONBOARDING.md` refer to
*"the Ministry of Home Affairs' official comparison tables"*. MHA's own page
(`mha.gov.in/en/commoncontent/new-criminal-laws`, which 403s to automated fetch
but loads in a browser) publishes **only the three bare acts — no correspondence
tables at all**. MHA published the statutes; its subordinate bodies published the
tables. There was never an MHA-hosted table to chase.

### Step 5 — Corpus citation analysis
Ran `core/ner.py` over all 7,130 judgments in `data/raw/summ` (IL-TUR SUMM,
207 MB). 75,184 citation rows.

Two correctness issues handled that would otherwise skew every number:
1. **Overlapping spans** — `ner.py` emits both a `STATUTE` ("Section 420 IPC")
   and a `PROVISION` ("Section 420") for the same text. Counting both
   double-counts. Provisions inside a statute span are dropped.
2. **Unattached provisions** — the onboarding's suggested *"nearest preceding
   STATUTE in the same paragraph"* heuristic is **inoperative on this corpus**.
   SUMM stores documents as *sentences* (median 126 chars) and the median gap
   back to an act mention is ~4,700 characters. A paragraph window attributed 4
   citations out of 821. Replaced with distance-based attribution, distance
   recorded per row, split into **strict** (≤600 chars) and **wide** so coverage
   can be reported honestly both ways.

### Step 6 — Registry schema
Implemented PLAN.md's three tables plus one deliberate addition.

**`SUPERSESSION` is a table, not a column.** PLAN.md puts
`superseded_by_section_id` on `sections` — implicitly 1:1. The real mapping is
many-to-many: BNS `1(1)–1(6)` replaces IPC §§1–5, and `BNS 179 ← IPC 237, 238,
239, 240, 241, 250, 251, 254, 258, 260, 489B`. A single FK would have kept IPC
237 and **silently dropped the other ten** — in the paper's most important
artifact. `superseded_by_section_id` is retained and populated only for
unambiguous 1:1 cases, so PLAN.md's schema and the M5 contract both still hold.

### Step 7 — Parser and ingestion
1,294 mappings parsed from the correspondence tables. Then discovered India Code
runs **DSpace 7 with a REST API**, giving one uniform ingestion path for every
statute — far better than bespoke PDF parsing. Ingested 2,088 sections with
official headings across 10 statutes.

### Step 8 — Resolver, check, tests
Built to the frozen contract. **22 tests: 21 passed, 1 xfailed.** Full repo
suite: 71 passed, no regressions.

### Step 9 — Heading backfill and section recovery
Recovered IPC/CrPC/IEA headings from the correspondence tables. Surfaced 14
sections present in the tables but absent from the registry (IPC 337, 338, 416,
431–436, 450, 451, 55; CrPC 4, 198A) and added them. Heading coverage
**62.6% → 100%**. Corrected the repeal annotation to distinguish a known absence
of successor from an unparsed mapping — see 5.17.

### Step 10 — The Constitution
Not in India Code (it is not an "Act" there), so sourced from the Legislative
Department's own published text. 466 articles, 445 headings, 21 marked omitted.
Required filtering an English-Kannada diglot by ASCII ratio (5.19) and a new
`status` column to record "omitted, date unknown" honestly (5.20). The
`xfail` guarding this gap is removed; two tests replace it.

### Step 11 — Resolution measurement
`resolution_rate.py` checks every corpus citation against the registry: 89.6%
resolve, and **34.7% of those are not in force today** (see 7b).

### Step 12 — Constitution omission dates
Second pass over the body recovered effective dates for 27 of 34 omitted
articles, turning `status='omitted'` markers into real in-force intervals.
Article 31 now has an exact boundary — in force 1979-06-19, repealed
1979-06-20 — making the Constitution a second dated natural experiment. The
remaining 7 are absent from the body entirely and keep an honest
"date not recorded".

### Step 13 — Section text and CITATION_STRINGS
India Code stores full statutory text in `dc.identifier.section_page_note` as
HTML; converted to text it fills `sections.text` for all ten India Code
statutes (2,075 sections). `CITATION_STRINGS` — the third table in PLAN.md's
schema and previously empty — now holds 19,784 distinct raw citation strings
with their targets and confidences.

---

## 5. Decisions, and why

Every non-obvious choice, with its reasoning. Several depart from the project
docs; each departure is argued rather than assumed.

### 5.1 Build in `M4/`, not `src/smartlawai/registry/`
`CLAUDE.md` §3 and `PLAN.md` put the registry at `src/smartlawai/registry/`;
`M4_ONBOARDING.md` §5 says top-level `registry/` and *"do not modify anything
under `src/smartlawai/`"* — a direct contradiction. The project owner chose a
self-contained `M4/` folder. **Why it works:** it satisfies the "don't touch
`src/`" constraint absolutely, creates zero merge surface with Track A, and the
M5 import is a one-line path change whenever integration happens. The cost is
that `gate.py` cannot `import smartlawai.registry` today — accepted deliberately,
because the frozen contract in `types.py` means the *shape* is fixed even though
the *location* is not.

### 5.2 NCRB HTML as the parse target, not the PDFs
Three sources carry the same correspondence data. **HTML won because it is
structurally unambiguous:** single `<table>`, no AJAX, and 100% of rows have
exactly 2 columns. The PDFs are two-column layouts that `pypdf` interleaves, and
their curly quotes return as mojibake — every row would need heuristic column
reconstruction, and every heuristic is a place to introduce silent errors into
the paper's ground truth. The PDFs are kept as **corroboration**, which is the
role they are actually good at.

### 5.3 Abandon the MHA bare-act PDFs for section lists
Downloaded and machine-readable (1.43M chars), so this was a real option. Rejected
after probing the layout: they are **Gazette publications**, with section headings
in a marginal column that `pypdf` extracts separately from the body, interleaved
with Devanagari transliteration, and kerned text that comes back as
`'THE BHARA TIY A NY A Y A SANHITA'`. Parsing that is bespoke per act.
India Code gives the same information as structured data through one API for
every statute. **Rule applied: prefer a structured source over a rendered one,
even when the rendered one is authoritative.** The bare acts are retained — they
remain the best source for section *text*, which is a later step.

### 5.4 `SUPERSESSION` as a table, not a column
The single most consequential schema decision. PLAN.md's
`superseded_by_section_id` is a single FK — implicitly 1:1. The data is
many-to-many: `BNS 179 ← IPC 237, 238, 239, 240, 241, 250, 251, 254, 258, 260,
489B`. A single column keeps IPC 237 and **silently discards the other ten**.
Silent truncation in the paper's ground-truth artifact is the worst available
failure mode — it produces confident wrong answers with no error. The column is
*retained* and populated only for unambiguous 1:1 mappings, so PLAN.md's stated
schema and the M5 contract both remain true; the table is the authoritative record.

### 5.5 Distance-based attribution, reported strict *and* wide
`M4_ONBOARDING.md` §4 suggests attributing a bare "Section 420" to the nearest
preceding statute *in the same paragraph*. Measurement killed it: SUMM stores
documents as **sentences** (median 126 chars), and the median gap back to an act
mention is **~4,700 characters**. The paragraph rule attributed 4 citations out
of 821. Widening the window recovers them but **attribution across 5,000
characters is a guess**, and I2 forbids presenting a guess as an answer. The
resolution is to keep both: record the distance on every row, and report a
**strict** number (≤600 chars, defensible) alongside a **wide** number (everything
attributed). One number would have been either dishonest or useless.

### 5.6 Structural compatibility as a hard constraint
`is_compatible()` refuses `art` for anything but the Constitution, refuses `s`
for the Constitution, and confines Orders/Rules to the CPC. **These are facts
about how the statutes are organised, not heuristics** — which is why they are
enforced as constraints rather than weighted as confidence. Before the guard, the
top-cited output contained `ipc-1860-art22` and `constitution-1950-s4`, neither
of which can exist. A rule that can be stated as a fact should be enforced, not
scored.

### 5.7 `resolve_citation` returns `None` rather than a low-confidence guess
`Section 9999 IPC` parses perfectly well and yields a canonical id — but no such
row exists. Two designs were possible: return the target and let
`check_citation` report `not_found`, or refuse. **Refusing is correct** because
the contract says *"normalise to a registry target"* — an id with no row behind
it is not a registry target. A `None` feeds the unresolved count, which is what
the coverage reporting needs, and `gate.py` strikes the claim either way.
`check_citation` still handles `not_found` defensively, so nothing fails open if
a caller constructs a citation by hand.

### 5.8 `check_citation` catches every exception
Deliberately broad, against normal Python style. Justified by I2: *"a verifier
that cannot run returns `status='unavailable'`, never a number"* — a safety
component must never fail open. If the registry file is missing, locked or
corrupt, the honest output is `not_found`, which strikes the claim. A propagated
exception would either crash the pipeline or, worse, be swallowed upstream and
treated as a pass.

### 5.9 Read-only database connection in the resolver
A serving process that resolves citations has no business mutating the registry.
Read-only makes that structural rather than conventional, and lets several
readers share the file.

### 5.10 `verified_by` left NULL on all 3,337 rows
The dates and mappings are drawn from official sources and are almost certainly
right. They are still recorded as **unverified**, because parsed is not verified
and threat T10 is specifically the risk of asserting authority we have not
checked — in the very artifact the project scores itself against. The registry
should never imply more confidence than it has earned.

### 5.11 Exhaustive coverage, but still report the coverage percentage
The project owner chose exhaustive over PLAN.md's demand-driven approach. Both
were kept: build exhaustively, **and** keep computing the coverage number. Full
coverage does not remove the need for the metric — it should improve it — and
threat T6's defence is *stating your own limit*, which requires the number
regardless of how good it gets. Measurement then vindicated the choice: 95%
coverage needs 1,566 sections, not the 100–150 PLAN.md predicts.

### 5.12 `xfail(strict=True)` for the Constitution instead of deleting the test
The Constitution is absent from the registry and is the most-cited authority in
the corpus. Deleting the test would hide that; leaving it failing would train
everyone to ignore a red suite. `strict=True` records the gap **and fails loudly
the moment it starts passing**, forcing the test back on when ingestion lands. A
companion test asserts the current behaviour is safe-if-incomplete: the citation
resolves to `None` and the gate strikes it, rather than passing through as valid law.

### 5.13 Rebuild the database rather than patch bad rows
`INSERT OR REPLACE` cannot remove rows that should never have existed — the 36
fabricated supersessions from the year-parsing bug. Rebuilding from scripts
guarantees the database matches the current code exactly, with no residue from
earlier buggy runs. This is safe **because the database is fully derived**: three
scripts reproduce it byte-for-byte from the source files.

### 5.14 Python 3.13 though the project targets 3.11
3.11 is not installed on this machine and `requires-python` is `>=3.11`. The risk
was `datasets<3.0` on 3.13; it resolved cleanly. **Recorded as a deviation**
because a future `requirements.lock` (invariant I8) must be produced on whichever
version is actually used for reported results.

### 5.15 No spaCy, no torch
`ner.py` degrades gracefully when spaCy is absent, and spaCy only contributes
PERSON/ORG/DATE/MONEY. `STATUTE` and `PROVISION` — the only labels M4 consumes —
come from the regex path. Skipping the `ml` extras avoided a multi-GB install and
a version-pin minefield the project docs describe as fragile, with **zero** loss
of capability for this milestone.

### 5.16 Recovering IPC/CrPC/IEA headings from the old-code column
India Code does not index the three repealed codes, so their headings had no
authoritative source and 1,249 sections sat heading-less. The correspondence
tables' *old-code* column carries them — `174A. Non-appearance in response to a
proclamation…` — so no new source was needed. These headings are tagged
`source_doc = 'ncrb-old-column'` to keep them **distinguishable from India Code
headings during verification**: they are weaker evidence and should be sampled
more heavily in the audit. Result: heading coverage went from 62.6% to 100%.

### 5.17 "Successor not recorded" is not "there is no successor"
Backfilling exposed 14 sections (IPC 337, 338, 416, 431–436, 450, 451, 55;
CrPC 4, 198A) that exist in the tables but had no *mapping* row, so no section
row was ever created. Adding them was straightforward. The subtler problem was
what to report: they had zero successors, and the code was annotating that as
*"no corresponding provision"* — the same words used for sedition, which was
genuinely deleted. **Those are different facts.** IPC 337 has a BNS counterpart;
we simply failed to parse it. Asserting otherwise states something unestablished,
which I2 forbids. The check now distinguishes an explicit `repealed_no_successor`
record ("no corresponding provision") from an absent one ("successor not recorded
in registry"). Both still strike the claim; only the honesty of the annotation
differs.

### 5.18 The Constitution: official PDF over convenient JSON
Machine-readable copies of the Constitution exist on GitHub in exactly the shape
this task wants. They were **not used**. `PLAN.md` requires *"primary sources
only … never secondary summaries"*, and threat T10 makes provenance load-bearing
precisely because this is the registry the project grades itself against.
Parsing an 799-page official PDF is more work than `curl`-ing a JSON file; that
is the cost of a defensible source, and it is the cheaper half of the trade.

### 5.19 Filter the diglot by ASCII ratio before parsing
The only reachable official edition is an English–Kannada diglot with the two
languages on **alternating pages**, Kannada in a legacy encoding. Parsing the
whole document let the Kannada column silently supply a heading for **Article 31**
when its English entry failed to match — a wrong value that looked like data,
not like an error. Filtering to English pages (≥92% ASCII alphabetic) before
matching eliminates the entire class. The general lesson: when two sources are
interleaved in one file, separate them *first*; do not rely on a pattern to
prefer the right one.

### 5.20 A `status` column, because "omitted" is not the same as "unknown date"
The arrangement of articles says **that** Article 31 was omitted, not **when** —
the date is in body footnotes. Leaving `in_force_to` NULL would report a
repealed article as good law; writing a guessed date would fabricate authority.
Neither is acceptable, so `SECTIONS` gained a nullable `status`. Omitted
articles carry `status='omitted'` with a NULL date, and are struck at **every**
as-of date — including 1960, because we genuinely cannot say when they stopped
applying. The registry records the limit of its own knowledge instead of
papering over it.

### 5.21 Two number-extraction traps worth remembering
Recovering omission dates hit two failure modes that look like data, not errors.
**Footnote superscripts glue onto article numbers** in PDF extraction: `3329A.`
is footnote 3 plus Article 329A. A magnitude rule (no article exceeds 395)
catches most, but not `132A` — plausible on its face, actually footnote 1 plus
Article 32A. Validating against the official article list settles it, because
Article 132A does not exist and 32A does. Second, **dates carry stray spaces**
(`w.e.f. 29-8- 1972`), so separators must tolerate whitespace. Both produced
silently wrong rows rather than visible failures — the reason the
cross-validation step exists at all.

### 5.22 Deleting a test when the data improves
`test_omitted_constitution_article_is_struck_without_inventing_a_date` asserted
Article 31 had no recorded date. That was true when written and false an hour
later, once the body pass recovered 1979-06-20. It was removed rather than
relaxed: two sharper tests replace it — one pinning the dated boundary, one
pinning the still-undated case (Article 238). A test that encodes a temporary
limitation should die when the limitation does, not be weakened into vagueness.

### 5.24 Keeping the unresolved citations in the table
My first `CITATION_STRINGS` loader filtered out `orphan` rows before writing.
The table then reported **0 unresolved** — a 100% resolution rate true only by
construction. That is exactly the flattering-but-false number this project
exists to argue against, produced in the project's own artifact. Orphans are
now kept with a NULL target, so 10,066 unresolved strings sit in the table
alongside 9,718 resolved ones and "flagged, never silently dropped"
(PLAN.md M4) is a queryable fact rather than a claim in a README.

### 5.25 Statute targets are targets too
The same loader initially grouped only on `section_id`, so bare act mentions
("Indian Penal Code" x711, "Constitution of India" x1313) fell into the
unresolved bucket. But `types.py` defines two target types, and a statute-level
citation resolves perfectly well. Fixing it moved 1,855 strings from
"unresolved" to "resolved as statute" — a reminder that a coverage number is
only as meaningful as the definition of "resolved" behind it.

### 5.23 Trusting the parser over my own ground truth
When the parser said IPC 124A had no successor and my test expected BNS 152, the
parser was right: the official table records sedition as **"Deleted"**, and BNS
152 is a distinct new offence. The general rule applied throughout: **when a
primary source and an expectation disagree, investigate the source before
"fixing" the code.** Three of my six original ground-truth entries were wrong.

---

## 6. Bugs found and fixed

| # | Bug | How found | Impact if shipped |
|---|---|---|---|
| 1 | **Act year parsed as section number.** `"Indian Penal Code, 1860"` → `ipc-1860-s1860` | Manual probe | Every bare act mention fabricated a phantom section |
| 2 | **Structurally impossible citations.** `ipc-1860-art22`, `constitution-1950-s4` | Reviewing top-cited output | The IPC has no Articles; the Constitution has no Sections. Fixed with `is_compatible()` |
| 3 | **Repeal marker only checked on one side.** Tables put "Deleted" in the *left* cell | Sedition ground-truth miss | Took `repealed_no_successor` from 1 row to 33 |
| 4 | **Commencement date not inherited.** BNS §318 reported `in_force` as of 2020 | Test failure | Sections inherited repeal dates but not start dates |
| 5 | **Years in cross-references parsed as sections.** `"...under section 84 of ... Sanhita, 2023"` → section `2023` | Cross-validation vs India Code | Created **36 false supersession rows** |
| 6 | **"No corresponding provision" asserted for merely-unparsed mappings.** IPC 337 has a BNS successor; the registry claimed it had none | Heading backfill | Stated an unestablished fact as authority — the exact I2 failure |

**On my own errors:** my first ground-truth test asserted IPC 124A → BNS 152 and
IPC 120 → BNS 61. The parser disagreed. **The parser was right and I was wrong** —
I had conflated 124 with 124A and 120 with 120B. The official table records
sedition (124A) as **"Deleted"**: BNS 152 is a distinct new offence, not a
renumbering. Corrected to 14/14.

---

## 7. Registry state

```
STATUTES        14
SECTIONS      3817   (3,809 headings 99.8%, 2,075 with full text 54.4%)
SUPERSESSION  1289   (deduplicated; 1,203 unambiguous 1:1)
CITATION_STRINGS 19784 (9,718 resolved, 10,066 flagged unresolved)
```

| Statute | Sections | Headings | Section-number source |
|---|---|---|---|
| ipc-1860 | 545 | ✓ | MHA tables only |
| crpc-1973 | 533 | ✓ | MHA tables only |
| bnss-2023 | 531 | ✓ | India Code (validated) |
| bns-2023 | 358 | ✓ | India Code (validated) |
| contract-1872 | 268 | ✓ | India Code |
| iea-1872 | 185 | ✓ (184) | MHA tables only |
| cpc-1908 | 171 | ✓ | India Code |
| bsa-2023 | 170 | ✓ | India Code (validated) |
| ni-1881 | 155 | ✓ | India Code |
| it-2000 | 125 | ✓ | India Code |
| consumer-2019 | 107 | ✓ | India Code |
| arbitration-1996 | 106 | ✓ | India Code |
| stamp-1899 | 97 | ✓ | India Code |

### Cross-validation
```
bns-2023    india code 358   registry 358   spurious 0   missing 0
bnss-2023   india code 531   registry 531   spurious 0   missing 0
bsa-2023    india code 170   registry 170   spurious 0   missing 0
```
**1,059 of 1,059 sections agree** across two independent sources.

### Working behaviour (as of 2026-09-05)
```
Section 420 IPC                          superseded  repealed 2024-07-01; see BNS 318
Section 302 of the Indian Penal Code     superseded  repealed 2024-07-01; see BNS 103
Section 124A IPC                         repealed    repealed 2024-07-01; no corresponding provision
Section 154 CrPC                         superseded  repealed 2024-07-01; see BNSS 173
Section 65B of the Indian Evidence Act   superseded  repealed 2024-07-01; see BSA 63
Section 318 BNS                          in_force
Section 9999 IPC                         UNRESOLVED  flagged; gate strikes the claim
```
Judged as of 2020-01-01, the same citations invert:
```
Section 420 IPC    in_force
Section 318 BNS    not_found   not in force until 2024-07-01
```

---

## 7b. Registry resolution over the corpus — the headline result

Every citation extracted from the 7,130-judgment corpus, checked against the
registry (`scripts/resolution_rate.py`). Invented section numbers are rejected
here rather than counted, so these figures supersede the earlier upper bound.

```
section-level citation instances : 27,167
resolved against registry        : 24,348 (89.6%)
unresolved (flagged, not dropped):  2,819 (10.4%)
```

| Act | Cited | Resolved | Rate |
|---|---|---|---|
| constitution-1950 | 12,394 | 12,147 | 98.0% |
| cpc-1908 | 4,855 | 3,506 | 72.2% |
| crpc-1973 | 4,436 | 3,930 | 88.6% |
| ipc-1860 | 4,110 | 3,613 | 87.9% |
| iea-1872 | 481 | 404 | 84.0% |
| it-2000 | 464 | 364 | 78.4% |
| contract-1872 | 354 | 318 | 89.8% |

**Authority status of resolved citations, as of 2026-09-05:**

| Status | Instances | Share |
|---|---|---|
| in_force | 15,844 | 65.1% |
| superseded | 7,497 | 30.8% |
| repealed | 1,007 | 4.1% |

> **8,504 of 24,348 resolved citations (34.9%) are not in force today, and
> entailment verification cannot detect a single one of them.**

That is the number `RESEARCH.md` §2 calls *"the number nobody reports"*,
measured rather than asserted. It is the natural experiment's core quantity and
it is now produced by a repeatable script.

---

## 8. Corpus findings

`PLAN.md` predicts *"typically 100–150 sections, not 511"* covers 95% of citation
mass. **Measured: 1,566.** The head matches the prediction (99 sections = 50%),
but the tail is far fatter — which supports the decision to build exhaustively.

| Citation mass | Sections needed |
|---|---|
| 50% | 99 |
| 80% | 574 |
| 90% | 1,058 |
| **95%** | **1,566** |
| 100% | 2,754 |

**The corpus is constitutional, not criminal.** Constitution 12,394 citations;
CPC 4,855; CrPC 4,436; IPC 4,110; IEA 481. The criminal three are **33.2%** of
attributed citations. `RESEARCH.md` §2 asserts *"Every Indian judgment before that
date cites IPC/CrPC/IEA"* — in this corpus they are a third, behind the
Constitution. 9,027 criminal citations is still ample for the experiment, but the
framing claim needs softening.

**58% of rows are orphans** (43,517) — citations to acts outside the seed set,
dominated by low-numbered sections (`section 3` ×1,250). Coverage must always be
reported against a stated denominator.

---

## 9. Known gaps and open items

| # | Item | Severity |
|---|---|---|
| 1 | ~~Constitution not ingested~~ — **RESOLVED**. 466 articles; 34 omitted, 27 with dated intervals. The other 7 have no date in the official text at all | Low |
| 2 | ~~IPC/CrPC/IEA headings missing~~ — **RESOLVED**. Recovered from the correspondence tables (100% coverage). Section *numbers* for these three remain single-sourced, so they still need heavier audit sampling | Medium |
| 3 | **No verification.** All 3,337 sections have `verified_by = NULL`. PLAN.md requires a 100-entry stratified hand-check plus a 200-entry independent audit (threat T10) | **High** |
| 4 | ~~Section text not ingested~~ — **RESOLVED** for the 10 India Code statutes (2,075 sections). IPC/CrPC/IEA and the Constitution have no text source; Constitution text is extractable from the PDF body if wanted | Low |
| 5 | Consumer Protection Act commencement seeded as `2020-07-20`; India Code records staged commencement incl. 24 July 2020 (S.O. 2421(E)) | Low |
| 6 | Coverage numbers are an **upper bound** — re-run once registry validation rejects invented sections | Medium |
| 7 | `M4/out/ic_*.json` are scratch from API discovery | Trivial |

**On the audit (item 3):** threat T10 requires the auditor be someone who did not
build the registry. The project owner directing this build is arguably inside
that boundary; the Track A owner would be a cleaner choice.

---

## 10. How to reproduce

```bash
cd C:\Users\ramch\smartlaw\M4\smartlawai-1
.venv\Scripts\python.exe M4\scripts\seed_statutes.py
.venv\Scripts\python.exe M4\scripts\load_registry.py
.venv\Scripts\python.exe M4\scripts\ingest_indiacode.py
.venv\Scripts\python.exe M4\scripts\validate_sections.py
.venv\Scripts\python.exe -m pytest M4\tests\test_registry.py -q
.venv\Scripts\python.exe M4\scripts\demo.py
```

`registry.duckdb` is fully derived — deleting it and re-running the first three
scripts rebuilds it exactly.

---

## 11. Compute

No GPU. No GCP. No paid API. Everything ran on CPU on a 14-core machine; the
heaviest job (NER over 7,130 judgments) took minutes. `CLAUDE.md` §2's hardware
routing is irrelevant to this milestone, exactly as `M4_ONBOARDING.md` states.
