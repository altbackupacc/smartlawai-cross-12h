# Your 100-Entry Check — How To Do It

**Your file:** `M4/out/verify_100.csv`
**Time:** 1–2 hours (fidelity check) or 5–8 hours (full legal check) — pick one below
**Output:** one word in the `VERDICT` column of every row

---

## What you are actually checking

The registry claims 1,289 statements of the form *"this old provision became
that new one on this date."* You are checking whether 100 of them are true.

Your sheet contains:

| | |
|---|---|
| 91 rows | `maps_to` — old section → new section |
| 9 rows | `repealed_no_successor` — old section died, nothing replaced it |
| 43 / 36 / 18 / 3 | IPC / CrPC / IEA / Constitution |

---

## First: pick your depth

**Do not do both. Decide now, write it at the top of the sheet.**

### Option A — Fidelity check (1–2 h)
*"Does the registry match the official table?"*

You compare each row against the government correspondence table the parser
read. Catches extraction bugs — which is the class that has actually bitten
this project. Does **not** catch an error in the government's own table.

### Option B — Legal check (5–8 h)
*"Is the mapping actually right?"*

You read both provisions and judge whether they correspond. Slower, needs legal
judgement, catches more.

**Recommendation: do Option A.** The independent 200-entry audit is the place
for Option B, and doing A first tells that auditor where to look.

---

## Setup — 10 minutes, once

Open these three things side by side:

1. **Your sheet** — `M4/out/verify_100.csv` in Excel or Google Sheets
2. **The source tables** — in `M4/registry/data/mha/`:
   - `ncrb_BNS.html` → for IPC rows (43 of yours)
   - `ncrb_BNSS.html` → for CrPC rows (36 of yours)
   - `ncrb_BSA.html` → for IEA rows (18 of yours)

   Open in a browser. **Ctrl+F is your main tool.**
3. **For the 3 Constitution rows** — `M4/registry/data/constitution/COI_english_2025-11.pdf`

Fill `CHECKED_BY` and `DATE` in row 2, then copy down the whole column. Do it
now so you don't forget at the end.

---

## The method — four steps per row

For each row, left to right:

1. Read `from_id` and `from_heading` — e.g. `ipc-1860-s307`, *Attempt to murder*
2. Open the matching source table, **Ctrl+F the old section number**
3. Read across to the new-code column
4. Compare with `to_id`. Type one word in `VERDICT`.

That's it. Most rows take 20–40 seconds.

---

## Three worked examples — real rows from your sheet

### V-001 — a normal mapping

```
from : crpc-1973-s342    Power to order costs
rel  : maps_to           effective 2024-07-01
to   : bnss-2023-s381    Power to order costs.
src  : ncrb_BNSS.html
```

Open `ncrb_BNSS.html`, Ctrl+F `342`. Find the row where the CrPC column shows
342. Does the BNSS column show 381?

- Yes → `correct`
- Shows a different number → `wrong`, and put the right number in `CORRECTED_VALUE`

Note both headings read "Power to order costs" — a matching heading is good
supporting evidence, but **the number is what you're verifying**.

### V-002 — high-impact, check carefully

```
from : ipc-1860-s307     Attempt to murder      (cited 59 times in the corpus)
to   : bns-2023-s109
```

Same method, in `ncrb_BNS.html`. This one is marked `1-critical` because it is
both single-sourced and heavily cited — an error here propagates further than
most. Slow down on the two critical rows.

### V-003 — a repeal with no successor

```
from : iea-1872-s113     Proof of cession of territory
rel  : repealed_no_successor
to   : (blank)
```

Here you're checking a **negative**: that the table really shows no counterpart.
In `ncrb_BSA.html`, find IEA 113. The BSA column should be empty, or say
*Omitted* / *Deleted* / *Repealed*.

- Empty or an omission marker → `correct`
- Shows a BSA section number → `wrong` (put that number in `CORRECTED_VALUE`)

**These 9 rows matter more than their count suggests.** Claiming a provision has
no successor when it does is the worse error — it tells a user the law vanished
when it merely moved.

---

## The three verdicts

| Write | When |
|---|---|
| `correct` | the source table says what the registry says |
| `wrong` | the source table says something different — **fill `CORRECTED_VALUE`** |
| `unclear` | you found the row but genuinely cannot tell, or couldn't find it |

**`unclear` is a real answer, not a cop-out.** It is excluded from the accuracy
figure rather than counted against the registry. Use it rather than guessing —
a guessed `correct` is worse than an honest `unclear`, because it inflates a
number the paper will publish.

**Never leave a row blank meaning "fine".** Blank counts as *unreviewed* and
makes the accuracy figure provisional.

---

## Things you will hit

**A heading that doesn't quite match.** Wording often changed between codes
(`Power to order costs` vs `Power to order costs.`). Judge on the **number**,
not the phrasing. Note the difference if it's substantial.

**Sub-sections.** The tables map at sub-section level (`1(3)`, `2(31)`); the
registry folds these to the parent section. So registry `crpc-1973-s46` may
correspond to several table rows for `46(1)`, `46(2)` etc. If any of them point
at the recorded target, that's `correct`.

**One-to-many.** Some old sections map to several new ones. If the table lists
multiple targets and `to_id` is one of them, that's `correct` — each pairing is
its own row in the registry.

**Constitution rows (3 of yours).** Different source: search the PDF's
arrangement of articles. An omitted article appears as `[31. Omitted.]`.

**Can't find the section at all.** Mark `unclear` and note "not found in table".
Don't hunt for more than a minute — that pattern is itself a finding.

---

## Pacing

Rows are **pre-sorted by priority**, so work top-down and never re-sort.

| Block | Rows | Time |
|---|---|---|
| `1-critical` (V-001, V-002) | 2 | 5 min — go slowly |
| `2-high` | 95 | 60–90 min |
| `3-medium` | 3 | 5 min |

Two sittings of about 45 minutes beats one long one — this is exactly the task
where attention drifts and errors creep in around row 60.

---

## When you're done

Save as CSV (keep the filename), then:

```bash
.venv\Scripts\python.exe M4\scripts\score_audit.py
```

It prints your verification accuracy and a per-priority breakdown. Once the
independent 200-entry audit is also filled in, the same command produces
Cohen's κ from the 30 shared rows.

Send me the finished sheet and I'll write the verdicts back into the registry —
rows you confirmed get `verified_by` and `verified_on` set, so the database
stops saying "nothing here has been checked."

---

## Two rules

**Do not compare notes with the 200-entry auditor.** 30 rows appear in both
sheets deliberately, and the agreement statistic only means something if the two
judgements are independent. Comparing first destroys the number and there is no
way to recover it afterwards.

**If you find yourself wanting to mark something `correct` because it's probably
fine — mark `unclear` instead.** The whole point of this registry is that it
refuses to state things it hasn't established. The verification should hold to
the same standard as the thing it's verifying.
