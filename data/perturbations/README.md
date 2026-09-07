# `data/perturbations/` — synthetic entailment training data

Training data for **M8a**'s InLegalNLI, not evaluation data. Built under M6
because it needs evaluation-split safety designed in from day one
(`M6_ONBOARDING.md` §9), and M6 is the milestone that owns what "an evaluation
split" means in this repo.

```bash
python data/perturbations/generate_perturbations.py --n 500
```

Writes `perturbations.jsonl` (gitignored — regenerate from this script plus its
fixed seed; do not commit the output) and `perturbations.manifest.json`
(the run's provenance: seed, counts, corpus, whether the train-split-only
restriction could be enforced).

## What it generates

Genuinely-entailed `(passage, claim)` pairs are read from IN-Abs (each
professionally-written headnote sentence paired with the judgment window it
overlaps most), then one of `RESEARCH.md` §2.5's seven legal perturbation types
is applied to produce a `not_entailed` counterpart:

| Type | Legal failure mode simulated |
|---|---|
| `section_number` | Wrong statutory provision |
| `ipc_bns_swap` | Repealed / superseded law |
| `negate_obligation` | Reversed legal effect ("shall" → "shall not") |
| `swap_parties` | Misattributed holding |
| `alter_date_amount` | Factual error |
| `change_court` | Wrong precedential weight |
| `unsupported_holding` | Fabricated ratio |

Type assignment is balanced across the applicable types per claim, so
`unsupported_holding` — the one type that can always fire — does not dominate
the set and per-type detection rates stay comparable.

## The two hard rules (`RESEARCH.md` §2.5, non-negotiable)

1. **Disjoint from every evaluation split.** Every row carries `source_doc_id`.
   `tests/test_perturbations.py` checks this against the shipped
   `eval/gold/*.jsonl` for real today (the generator reads IN-Abs
   **train-data**; the seed gold sets are built from **test-data**; the two
   folders share zero document ids). Once `data/splits/train.json` exists, the
   same test file enforces that every source document is drawn from the
   **training** split specifically — currently `pytest.skip`-guarded with a
   stated reason, same convention as `M1_ONBOARDING.md` §9's own leakage test.
2. **Never generated from this project's own system outputs.** Only from
   genuine IN-Abs `(passage, claim)` pairs — otherwise the judge would learn to
   like this project's writing style instead of detecting real faithfulness
   failures.

## Releasable standalone

`generate_perturbations.py` has no imports outside the Python standard library
and no dependency on this repo's own packages (`tests/test_perturbations.py`
asserts this). Copy the file plus the IN-Abs corpus and it runs anywhere —
`RESEARCH.md` §2.5 requires the generator itself to be publishable, not just
its output.

## What "genuinely entailed" means here, precisely

A headnote sentence is paired with the judgment window it shares the most
content words with, and kept only when that overlap clears `--min-overlap`
(default 0.35). This is **a proxy for entailment, not a proof of one** — every
row records its overlap score so a stricter filter can be applied downstream
without regenerating, and the threshold used is recorded in the run manifest.
