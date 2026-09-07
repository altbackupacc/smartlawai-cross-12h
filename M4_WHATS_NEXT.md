# M4 — What's Left

The registry is built and the code is done. Three things remain, and none of
them are engineering.

**Where everything lives:** `C:\Users\ramch\smartlaw\M4\smartlawai-1\M4\`

---

## The short version

| # | Task | Who | Rough effort |
|---|---|---|---|
| 1 | Hand-verify 100 mappings against the Gazette | You (or anyone) | 4–6 h |
| 2 | Independent audit of 200 mappings | **Someone who did not build this** | 8–10 h |
| 3 | Publish the registry | You | 1–2 h |

Everything needed for 1 and 2 is already generated and waiting.

---

## 1. The 100-entry hand-check

**File:** `M4/out/verify_100.csv` — open in Excel or Sheets.

Each row is one supersession mapping the registry claims. Your job is to say
whether it is right.

**Fill in three columns:**

| Column | What to put |
|---|---|
| `VERDICT (correct / wrong / unclear)` | one of those three words |
| `CORRECTED_VALUE` | only if wrong — what it should be |
| `NOTES` | anything worth remembering |

Also fill `CHECKED_BY` and `DATE` once at the top and copy down.

**How to check one row.** Take `from_id` (e.g. `ipc-1860-s420`) and `to_id`
(`bns-2023-s318`). Confirm against a primary source that IPC 420 really does
correspond to BNS 318. Use the Gazette, or the official correspondence tables
in `M4/registry/data/mha/`. `correct` means the mapping is right; `wrong` means
it points somewhere else; `unclear` means the sources genuinely don't settle it
— that is a legitimate answer and it is excluded from the accuracy figure
rather than counted against the registry.

**Rows are sorted by priority — work top-down.** If you run out of time, the
most important ones are already done.

| Priority | Meaning |
|---|---|
| `1-critical` | single-sourced **and** heavily cited — an error here does the most damage |
| `2-high` | single-sourced (IPC / CrPC / IEA — no second source exists for these) |
| `3-medium` | cross-validated, but unusual shape (no successor, or one-to-many) |

---

## 2. The 200-entry independent audit

**File:** `M4/out/audit_200.csv` — same columns, same method.

**This one cannot be done by whoever built the registry.** That is threat T10
in `RESEARCH.md`: the project scores itself against this registry, so an
independent check is the whole defence. The Track A owner is the natural choice.

**30 rows appear in both sheets on purpose.** That is not duplication — it is
how the agreement statistic gets computed. The two reviewers must fill them
**independently, without comparing notes**, or the number is meaningless.

---

## 3. Score it

Once both sheets have their `VERDICT` column filled:

```bash
.venv\Scripts\python.exe M4\scripts\score_audit.py
```

That prints the two numbers the paper needs:

- **verification accuracy** — `PLAN.md` M4 step 4 ("report verification accuracy
  in the paper")
- **Cohen's kappa** — `RESEARCH.md` §7.5 ("report agreement for every study")

It also lists any rows the two reviewers disagreed on. **Resolve those by
discussion, not by averaging.**

Then write the confirmed verdicts back: rows judged `correct` get `verified_by`
and `verified_on` set. Right now all 3,817 rows have `verified_by = NULL`, which
is the registry honestly saying nothing in it has been checked.

---

## 4. Publish the registry

`PLAN.md` M4 lists this and nobody has done it: *"Publish the registry — it is a
standalone contribution."* It is currently a local DuckDB file. Publishing means
exporting to something citable (CSV/JSON plus a schema description and the
provenance docs) and putting it somewhere with a stable link.

Do this **after** verification, so what gets published carries the accuracy
figure with it.

---

## Where the numbers came from

Sampling isn't uniform. It is weighted toward rows where an error costs most:

- **Single-sourced rows are over-sampled.** IPC/CrPC/IEA section numbers come
  from one source only; everything else was cross-checked against India Code.
- **Unusual shapes get a guaranteed share** — repealed-with-no-successor, and
  one-to-many mappings — because that is where parsing errors concentrate.
- **Frequently-cited sections are prioritised.** A wrong mapping on a section
  cited 400 times in the corpus matters more than one never cited at all.

The draw is seeded (`SEED = 20260905` in `make_audit_sample.py`), so anyone who
doubts the sample can regenerate exactly the same one.

---

## Known limits — read before auditing

These are places the registry is thinner than it looks. They are not bugs; they
are the honest edges of the sources.

| | |
|---|---|
| **IPC / CrPC / IEA rest on one source** | India Code does not index the repealed codes at all. Everything else has two independent sources. Weight these heavier. |
| **No full text for those three, or the Constitution** | Headings only. The other ten statutes have complete section text. |
| **7 Constitution articles have no repeal date** | 31D, 131A, 226A, 228A, 238, 259, 272 appear only in the arrangement of articles; the date is nowhere in the official text. They are struck at every date with a note saying exactly that. |
| **Consumer Protection Act** | Seeded as commencing 2020-07-20, but it commenced in **stages** — India Code records a block of provisions on 24 July 2020. Worth one row of attention. |
| **`arbitration-1996`** | 93 of 106 sections have text; 13 came back empty from India Code. |

---

## One thing worth knowing about how this was built

The whole registry was built in a single session by one agent, with no review.
Five real bugs surfaced **only** because a second source contradicted the first
— including one that fabricated 36 supersession mappings from years mistaken for
section numbers, and one where the parser was right and the hand-written test
was wrong.

That is an argument for taking the audit seriously, not for trusting the speed.
The cross-validation that exists (1,059/1,059 sections agreeing across two
sources for BNS/BNSS/BSA) covers the *new* codes. The repealed ones — the
paper's actual centrepiece — have no such second source. That is exactly what
the 200-entry audit is for.
