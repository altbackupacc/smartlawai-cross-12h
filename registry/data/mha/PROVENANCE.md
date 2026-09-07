# Supersession correspondence tables — provenance log

`PLAN.md` M4 requires **primary sources only**. Threat T10 makes sourcing
load-bearing: we are building the registry the project scores itself against,
so every row must be traceable, and where possible corroborated by a second
independent official source.

## The week-1 unknown: RESOLVED — all tables are machine-readable

`PLAN.md`: *"are those tables machine-readable? Parseable → saves ~20 h.
Scanned images → costs ~20 h. Largest single unknown in the whole estimate."*

**Every table is machine-readable. None require OCR.** No Tesseract or poppler
install is needed; the ~20h OCR branch is closed.

## Sources held

| Mapping | File | Source | Format | Size |
|---|---|---|---|---|
| BNS ↔ IPC | `ncrb_BNS.html` | cytrain.ncrb.gov.in | HTML, 600 rows × 2 cols | 764 KB |
| BNS ↔ IPC | `BNS_IPC_Comparative_uppolice.pdf` | uppolice.gov.in | PDF, 23 pp, 66,591 chars | 1.09 MB |
| BNSS ↔ CrPC | `ncrb_BNSS.html` | cytrain.ncrb.gov.in | HTML, 636 rows × 2 cols | 812 KB |
| BNSS ↔ CrPC | `BNSS_CrPC_comparison_bprd.pdf` | bprd.nic.in | PDF, 38 pp, 75,644 chars | 675 KB |
| BSA ↔ IEA | `ncrb_BSA.html` | cytrain.ncrb.gov.in | HTML, 209 rows × 2 cols | 260 KB |
| BSA ↔ IEA | `BSA_IEA_comparison_bprd.pdf` | bprd.nic.in | PDF, 14 pp, 28,511 chars | 393 KB |

All retrieved 2026-09-05.

### Why these count as primary

Both hosts are **Ministry of Home Affairs bodies**, on official government
domains:

- **NCRB** — National Crime Records Bureau, an MHA organisation. `cytrain.ncrb.gov.in`
  is its official training portal.
- **BPRD** — Bureau of Police Research & Development, an MHA organisation.
  `bprd.nic.in` is on NIC (National Informatics Centre) infrastructure.

`uppolice.gov.in` is a **state** police site and is retained only as a third
corroborating copy, never as the sole source for any row.

### Corroboration

Every mapping has **at least two independent official sources**. The NCRB BNS
HTML and the UP Police BNS PDF were spot-checked against each other and agree
exactly on the opening rows (BNS 1(1)→IPC 1; 1(2) New Section; 1(3)→IPC 2;
1(4)→IPC 3; 1(5)→IPC 4; 1(6)→IPC 5; 2(1) "act"→IPC 33; 2(2) "animal"→IPC 47).
A full programmatic diff of HTML vs PDF is the intended cross-check and is the
strongest available answer to T10 short of the Gazette itself.

### The MHA page itself — RESOLVED, and it changes the picture

`mha.gov.in/en/commoncontent/new-criminal-laws` 403s to automated fetch but
loads normally in a real browser. Retrieved 2026-09-05.

**MHA hosts no correspondence tables.** Its New Criminal Laws page publishes
exactly three documents — the three new bare acts — and nothing else. So the
premise in `PLAN.md` and `M4_ONBOARDING.md` ("the Ministry of Home Affairs'
official comparison tables") is imprecise: MHA published the *statutes*; the
*correspondence tables* were published by MHA subordinate bodies (NCRB, BPRD)
and by state police forces.

There is therefore no MHA-hosted correspondence table to chase. **NCRB and BPRD
are the closest to source that exists**, and the provenance question is closed
on the best available terms rather than left open.

### Bare acts — retrieved from MHA directly

| File | Source URL | Pages/chars | Verdict |
|---|---|---|---|
| `MHA_BNS_2023_bare_act.pdf` | mha.gov.in/.../250883_english_01042024.pdf | 407,775 chars | MACHINE-READABLE |
| `MHA_BNSS_2023_bare_act.pdf` | mha.gov.in/.../250884_2_english_01042024.pdf | 857,117 chars | MACHINE-READABLE |
| `MHA_BSA_2023_bare_act.pdf` | mha.gov.in/.../250882_english_01042024_0.pdf | 169,980 chars | MACHINE-READABLE |

1.43M characters of statutory text at primary-source quality. These give the
**authoritative section lists and section text for BNS, BNSS and BSA**, which:

- validates the section rows parsed out of the correspondence tables (any
  `bns-2023-sNNN` not present in the bare act is a parse error, not a section);
- supplies `sections.heading` and `sections.text`, currently NULL;
- covers 3 of the 10 seed statutes for the exhaustive build at the highest
  possible provenance.

The remaining seven (IPC, CrPC, IEA, Constitution, Contract Act, Arbitration,
Consumer Protection, CPC, NI Act, IT Act, Stamp Act) still need India Code.

## Parsing notes

**Use the NCRB HTML as the parse target**, not the PDFs. All three are static
HTML (no AJAX), single `<table>`, and 100% of rows have exactly 2 columns. The
PDFs are two-column layouts that `pypdf` interleaves and whose curly quotes come
back as mojibake — usable for cross-validation, painful as a primary parse.

Row shapes the parser must handle, all observed in the data:

| Shape | Example | Meaning |
|---|---|---|
| Header | `['Bharatiya Nyaya Sanhita, 2023', 'Indian Penal Code, 1860']` | skip |
| Chapter | `['CHAPTER I – PRELIMINARY', 'CHAPTER I – INTRODUCTION']` | skip |
| Normal | `['1(3)', '2. Punishment of offences...']` | BNS 1(3) ← IPC 2 |
| New provision | `['1(2)', 'New Section']` | no predecessor |
| **Repealed, no successor** | `['', '2. Repealed']` | IEA §2 died with nothing replacing it |
| One-to-many | `['2. Definitions.', '3. Interpretation-clause. 4. ― May Presume...']` | BSA 2 ← IEA 3 **and** 4 |

The last two are why `SUPERSESSION` is a table rather than a column on
`sections` (see `registry/schema.py`). A single `superseded_by_section_id`
cannot express "replaced by nothing" or "replaced by three sections".
