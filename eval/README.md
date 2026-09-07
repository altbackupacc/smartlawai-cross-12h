# `eval/` — M6: evaluation harness + gold sets

Scoring for the whole project, plus the human-labelled test sets that scoring
depends on. `PLAN.md` §M6, `RESEARCH.md` §5–7 and `M6_ONBOARDING.md` are the
specification; this file is a map of where each piece landed.

See `docs/M6_IMPLEMENTATION_ANALYSIS.md`, `docs/M6_IMPLEMENTATION_PLAN.md` and
`docs/M6_COMPLETION_REPORT.md` for the design, the phase-by-phase plan, and
what actually shipped.

## Layout

```
eval/
├── config.py           every threshold/path M6 uses (CLAUDE.md §4 applied to eval)
├── schemas.py           the five gold-set dataclasses + the annotation record
├── gold_sets.py         load / validate / checksum / power-calculation
├── metrics.py            retrieval, groundedness, citation, refusal, judge, cost metrics
├── extractiveness.py    4-gram copying measure -- mandatory alongside faithfulness
├── stats.py              paired bootstrap, ECE, reliability data, Bonferroni
├── report.py             results dict -> JSON + human-readable Markdown
├── run_eval.py           the orchestrator; the ONLY module that imports smartlawai
├── build_seed_gold.py    builds the seed gold sets shipped in eval/gold/
├── annotate/
│   ├── guidelines.py     rubrics and rating scales, as versioned data
│   ├── store.py          append-only annotation log + export to eval/gold/*.jsonl
│   ├── agreement.py      Cohen's kappa / Krippendorff's alpha, live and standalone
│   ├── pooling.py        TREC pooling for the retrieval set's candidates
│   └── app.py             the Streamlit annotation tool
├── gold/                 the five gold sets (frozen, committed) -- see gold/PROVENANCE.md
└── results/              run_eval.py output -- committed, kept forever (OPS.md §4)
```

## Run it

```bash
pip install -e ".[eval,ui,dev]"
python -m eval.run_eval --split dev
```

Writes `eval/results/dev_<timestamp>.json` and `.md`. With every model still
stubbed (M0), the table is full of trivial and `unavailable` entries — that is
the M6 done-when criterion (`M6_ONBOARDING.md` §2): the table computes end to
end and is honest about what it could not measure.

Compare against the last committed baseline (the CI gate does this on every PR
touching `eval/`, `src/smartlawai/`, or `sample_docs/`):

```bash
python -m eval.run_eval --split dev --baseline eval/results/dev_<earlier>.json
```

## Annotate

```bash
streamlit run eval/annotate/app.py
```

Pick a gold set and enter a rater id. Randomised order, blinded system labels,
~10% cold subset for the anchoring control, live agreement in the sidebar, and
a one-click export back to `eval/gold/*.jsonl` — all described in
`eval/annotate/app.py`'s own docstring and enforced by `tests/test_annotation_store.py`.

## Build (or rebuild) the seed gold sets

```bash
python -m eval.build_seed_gold
```

**Read `eval/gold/PROVENANCE.md` before trusting a number from these files.**
They are small, honestly-labelled seed sets — not the 300–400-item, multi-rater
study `RESEARCH.md` §7 specifies. Every row says how its label was produced.

## The one rule that runs through all of it (CLAUDE.md I2)

A metric that could not be computed returns `None` with a stated reason, never
a substituted `0.0`. `eval/report.py` renders that as `unavailable`, and lists
every such metric in its own report section, so "not measured" and "measured
zero" are never confusable in a table or in the paper.
