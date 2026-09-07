# M4 — drop-in package

You already have the `smartlawai` repo. This adds the M4 authority registry.

## Install

Unzip so that `M4/` sits at the **repo root**, next to `src/`, `tests/`,
`pyproject.toml`:

```
smartlawai-1/
├── src/
├── tests/
├── pyproject.toml
└── M4/          <-- this package
```

Nothing outside `M4/` is touched or replaced. M4 is self-contained by design —
it does not import from `src/smartlawai/` except in two corpus scripts (noted
below), so there is no merge surface with your work.

## Build the registry (~3 min, needs network for India Code)

```bash
.venv\Scripts\python.exe M4\scripts\seed_statutes.py
.venv\Scripts\python.exe M4\scripts\load_registry.py
.venv\Scripts\python.exe M4\scripts\ingest_indiacode.py
.venv\Scripts\python.exe M4\scripts\backfill_headings.py
.venv\Scripts\python.exe M4\scripts\ingest_constitution.py
.venv\Scripts\python.exe M4\scripts\load_citation_strings.py
```

Then confirm:

```bash
.venv\Scripts\python.exe -m pytest M4\tests\test_registry.py -q     # 24 passed
.venv\Scripts\python.exe M4\scripts\demo.py
```

`M4/out/registry.duckdb` is **not shipped** — it is fully derived from the six
scripts above and the source documents in `M4/registry/data/`. Rebuilding it
locally is the point: if your build doesn't match, that's a finding.

Requirements: Python 3.11+ with `duckdb`, `pandas`, `pypdf`, `pytest`
(`pip install -e ".[dev,data]"`). **spaCy and torch are not needed.**

## What it gives you

```
14 statutes · 3,817 sections · 1,289 supersessions · 19,784 citation strings
```

A resolver and an authority check, to a frozen contract in
`M4/registry/types.py`:

```python
resolve_citation(raw_text) -> ResolvedCitation | None
check_citation(citation, as_of) -> RegistryResult   # never raises
```

`M5`'s `gate.py` treats every non-`in_force` status the same way: it strikes the
claim. `not_found` is a valid, honest return (CLAUDE.md I2).

## Read these first

| File | |
|---|---|
| `M4/M4_WORKLOG.pdf` | what was built, every decision and why, all known limits |
| `M4/WHATS_NEXT.pdf` | the three remaining tasks |
| `M4/registry/data/*/PROVENANCE.md` | where every source came from |

## Two things to know

**Nothing is verified.** All 3,817 rows carry `verified_by = NULL`. The
100-entry hand-check and 200-entry independent audit are outstanding — worksheets
are in `M4/out/`.

**The two corpus scripts need things not in this package.**
`scripts/extract_citations.py` and `scripts/inspect_corpus.py` import
`smartlawai.core.ner` and read `data/raw/summ` (207 MB, not shipped). Their
output is already cached in `M4/out/citations.parquet`, so nothing downstream
needs re-running them.
