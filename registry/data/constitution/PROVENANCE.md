# Constitution of India — provenance

**Source:** `THE CONSTITUTION OF INDIA [As on 11th November, 2025]` —
Government of India, **Ministry of Law and Justice, Legislative Department**,
via `legislative.gov.in`. Retrieved 2026-09-05.

| File | Bytes | Pages | Verdict |
|---|---|---|---|
| `COI_english_2025-11.pdf` | 6,655,162 | 799 | MACHINE-READABLE (1,939,303 chars, 2,427/page) |

## Why not India Code

**The Constitution is not in India Code.** It is not an "Act" in that database's
model and does not appear in the `ACT` collection — confirmed by title search
across all collections. Since India Code is the source for the other ten
statutes, this one needed a different primary source, and the Legislative
Department's own published text is it.

## Why not the GitHub JSON copies

Several third-party machine-readable copies of the Constitution exist
(`civictech-India/constitution-of-india`, `Yash-Handa/The_Constitution_Of_India`).
They were **deliberately not used**. `PLAN.md` M4 requires *"primary sources
only — Gazette of India, eSCR — never secondary summaries"*, and threat T10
makes sourcing load-bearing because this is the registry the project scores
itself against. Convenience does not outrank provenance here.

## Edition caveat — read before trusting a heading

The retrieved edition is an **English–Kannada diglot**: English and Kannada
alternate *page by page*, and the Kannada is in a legacy encoding that extracts
as Latin-1 mojibake (`©lÄÖ©qÀ¯ÁVzÉ`).

This is not cosmetic. On a first parse, **Article 31's heading came back as
Kannada text**, because its English entry did not match the pattern and the
Kannada page silently supplied one. The parser therefore filters to English
pages by ASCII ratio (≥92% of alphabetic characters ASCII) *before* matching.
After filtering, zero Kannada leaked; the only non-ASCII remaining in headings
are legitimate curly apostrophes ("President's office", "Money Bills").

A plain English-only edition would be preferable. `lddashboard.legislative.gov.in`
was unreachable (connection failure, not 403) and
`legislative.gov.in/document/constitution-of-india-in-english` does not resolve
to a document page. **If a monolingual English edition becomes reachable, re-run
the ingest against it and diff** — that is a cheap, high-value check.

An English–Gujarati edition was downloaded first by mistake
(`316140f5938919bbbce4c22f1bf3d5ff.pdf`) and removed; it is recorded here only
so nobody repeats the error.

## What is parsed

The **arrangement of articles** (front matter), not the body. Two entry shapes:

```
32. Remedies for enforcement of rights conferred by this Part ... 19
[31. Omitted.] ................................................. 16
```

The bracketed form is the official marker for a repealed/omitted article.

**Result: 466 articles, 445 with headings, 21 marked omitted.**

The 21 omitted articles:

```
2A  31  31D  32A  131A  144A  226A  228A  238  242  257A
259  268A  272  278  291  306  314  329A  359A  362
```

These correspond to known repeals — Art 2A (Sikkim, 36th Amendment), Art 31
(right to property, 44th), Art 291 (privy purses, 26th), Art 329A (the election
article, 44th), and the 42nd/43rd Amendment group.

## The honesty problem, and how it is recorded

The arrangement of articles says **that** an article was omitted but not
**when** — the amendment and effective date sit in body footnotes. Three options
existed, and two were unacceptable:

| Option | Verdict |
|---|---|
| Leave `in_force_to` NULL | ✗ reports a repealed article as good law |
| Write a guessed date | ✗ fabricates authority — exactly what I2 forbids |
| Record the fact without the date | ✓ |

So `SECTIONS` gained a nullable `status` column. Omitted articles carry
`status='omitted'` with `in_force_to` NULL, and `check_citation` strikes them at
**every** as-of date with the note *"omitted from the Constitution; effective
date not recorded in registry"*. We cannot claim it was in force in 1960 either,
because we do not know when it stopped being so.

## Omission dates — recovered

A second pass parses the **body** entries, which carry both the original heading
and the effective date:

```
31.  [Compulsory acquisition of property.].— Omitted by the Constitution
     (Forty-fourth Amendment) Act, 1978, s. 6 (w.e.f. 20-6-1979).
291. [Privy purse sums of Rulers.].— Omitted by the Constitution
     (Twenty-sixth Amendment) Act, 1971, s. 2 (w.e.f. 28-12-1971).
```

**Result: 34 omitted articles, 27 with a dated in-force interval, 7 date-unknown.**
The body pass also found omissions the arrangement of articles alone did not
list (Arts. 379–391, removed by the Seventh Amendment in 1956), so it improved
completeness as well as dating.

Two extraction traps, both real:

- **Footnote superscripts glue onto article numbers.** `3329A.` is footnote 3
  followed by Article 329A. A magnitude check (no article exceeds 395) catches
  most, but not `132A` — a plausible-looking number that is really footnote 1 +
  Article 32A. Resolved by preferring a candidate that appears in the official
  arrangement of articles; Article 132A does not exist, Article 32A does.
- **Dates carry stray spaces** — `(w.e.f. 29-8- 1972)` — so the separators must
  tolerate whitespace.

The remaining **7 (31D, 131A, 226A, 228A, 238, 259, 272)** appear *only* in the
arrangement of articles as `[NNN. Omitted.]` and are absent from the body
entirely — the official text drops fully-omitted articles rather than leaving a
stub. **Their dates are not in this document at all**, so they keep
`status='omitted'` with no date and are struck at every as-of date. That is the
honest floor, not a parser failure.

### The result this produces

Article 31 now carries a real interval, making the Constitution a **second dated
natural experiment** alongside IPC→BNS, exactly as `PLAN.md`'s V2 note
anticipated — and with no schema change:

```
as of 1960-01-01 : in_force
as of 1979-06-19 : in_force
as of 1979-06-20 : repealed   repealed 1979-06-20; no corresponding provision
as of 2026-09-05 : repealed   repealed 1979-06-20; no corresponding provision
```

## Why this matters to the paper

Article 31 alone is cited **443 times** in the corpus. It was repealed in 1979.
Any system retrieving pre-1979 case law and answering a present-day property
question will cite it — grounded, entailed, and no longer law. That is the
project's thesis reproduced in a second, independent statute.
