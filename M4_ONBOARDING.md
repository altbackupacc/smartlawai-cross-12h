# M4 Onboarding — Authority Registry

You're picking up **M4** on `smartlawai` while someone else works the rest of the
system in parallel. This doc is meant to be everything you need to start in a fresh
Claude Code chat without re-deriving context from anyone. Read it top to bottom once,
then use it as a reference.

**No GPU, no cloud needed for any of this.** M4 is pure Python + a local database.
Don't touch Cloud Shell, `gcloud`, or Docker for this work — none of it applies here.

---

## 1. Read these four files in the repo root first, in this order

1. **`CLAUDE.md`** — non-negotiable invariants (especially **I1**, **I2**), repo layout,
   conventions. Short, read it fully.
2. **`PLAN.md`** — the milestone tracker. Find the `## M4` section — that's your literal
   spec: schema, sourcing strategy, deliverables, and the exact "done when" test.
3. **`RESEARCH.md`** — §1–2 explain *why* the registry is the paper's whole
   contribution (not a supporting feature), and describe the IPC→BNS natural
   experiment your registry directly enables. Worth understanding before you write
   a line of code — it tells you which parts of the schema actually matter.
4. **`OPS.md`** §8 ("Parallel Work") — this *is* the two-person plan someone already
   wrote for exactly this situation. It's why M4 was picked as your track: fully
   standalone, no dependency on the system code the other person owns.

---

## 2. What you're actually building (summary of `PLAN.md`'s M4)

A database of Indian statutes and sections, with in-force intervals and supersession
mappings, plus a resolver that takes a citation string and tells you whether it's
currently valid law.

**Schema** (DuckDB — see §5 below for why):
```sql
statutes(id, short_title, long_title, year, jurisdiction,
         in_force_from, repealed_on, repealed_by_statute_id)
sections(id, statute_id, number, heading, text,
         in_force_from, in_force_to, superseded_by_section_id)
citation_strings(id, raw, normalised, target_type, target_id, confidence)
```

**Build demand-driven, not exhaustively.** Extract citations from whatever corpus
exists (see §4), rank by frequency, and cover the top ~95% of citation mass — not all
511 IPC sections. Report the coverage percentage explicitly; unresolved citations get
flagged, never silently dropped.

**The centerpiece**: the IPC→BNS, CrPC→BNSS, IEA→BSA supersession mapping (effective
2024-07-01), sourced from the Ministry of Home Affairs' official comparison tables —
parsed programmatically, then hand-verified on a stratified sample of 100 entries.
**Check in week one whether those tables are machine-readable or scanned images** —
`PLAN.md` calls this "the largest single unknown in the whole estimate." If they're
scanned, that's a ~20h cost instead of ~2h, and changes your whole week's plan — flag
it immediately rather than pushing through alone.

**Done when**: `test_repealed_ipc_section_is_struck` passes — a claim citing IPC §420
with an as-of date of today gets struck and annotated `"repealed 2024-07-01; see BNS
§318"`.

---

## 3. Seed statutes (from `PLAN.md`)

IPC 1860 → **BNS 2023** · CrPC 1973 → **BNSS 2023** · IEA 1872 → **BSA 2023** ·
Indian Contract Act 1872 · Arbitration & Conciliation Act 1996 ·
Consumer Protection Act 2019 · CPC 1908 · NI Act 1881 · IT Act 2000 · Indian Stamp Act
1899

Source from **primary sources only** — Gazette of India, eSCR — never secondary
summaries. This is load-bearing for threat T10 in `RESEARCH.md` (you're building the
registry the project scores itself against; primary sourcing plus an independent
audit is the whole defense).

---

## 4. What already exists — reuse it, don't rebuild it

- **`src/smartlawai/core/ner.py`** — a working regex + spaCy citation extractor. It
  already recognizes IPC/CrPC/BNS/BNSS/Contract Act/Constitution/etc. section
  references in several phrasings ("Section 420 IPC", "Section 302 of the Indian
  Penal Code"). This is your input for the demand-driven frequency analysis — run it
  over whatever corpus exists and `Counter()` the results. It has 13 passing unit
  tests in `tests/test_ner.py` already.
- **One known limitation to design around**: a bare section number mentioned without
  the act name nearby (e.g. "Section 420" appearing two sentences after the Act was
  last named) comes back as a generic `PROVISION` entity with no statute attached.
  Your resolver needs a fallback for this — the natural one is attributing an
  unattached `PROVISION` to the nearest preceding `STATUTE` mention in the same
  paragraph — since undercounting per-statute citation frequency would quietly
  deflate your coverage number.
- **`src/smartlawai/adapters/local.py`** — the existing DuckDB pattern to follow:
  named-column inserts (never positional — see `CLAUDE.md` §4), `CREATE TABLE IF NOT
  EXISTS`, this general shape. Don't reinvent the storage style.
- **`pyproject.toml`'s base `dependencies`** already include `duckdb>=0.10` — you
  don't need to add anything for the registry database itself.
- **Corpus availability**: M1 (data foundation) may or may not be done yet when you
  start — check with whoever owns Track A/B. If it isn't, use
  `sample_docs/sample_judgment.txt` for structural testing only (proving your
  extraction/resolution pipeline works end to end), not for real coverage numbers —
  those need the real corpus.

---

## 5. Where your code goes — and what NOT to touch

**Build only in these paths** (none of them exist yet — you're creating them):
- `registry/` — schema, resolver, the MHA table parser, seed data
- `verify/registry_check.py` — the consumer-facing check function (§6 below)
- `tests/test_registry.py`

**Do not modify**: `gcloud/`, `train/`, `docker/`, `.github/`, anything already under
`src/smartlawai/`, or `.env`/`.env.example`. These are actively owned and working —
the GCP training pipeline in particular took a full session of debugging to get
running end to end. Touching them risks both merge conflicts and breaking something
that currently works. If you genuinely need a change there, say so explicitly rather
than editing directly.

**If you need a new dependency**, add a new `registry` extras group to
`pyproject.toml` rather than touching the existing `ml`/`train` groups — those have
deliberate upper-bound version pins (read the comments next to them) that took
several rounds of debugging a live GPU training job to get right. Don't loosen them.

---

## 6. The interface contract — this is what prevents incompatibilities

The other person will build `gate.py` (M5) later, which imports and calls your code
directly. Expose exactly this shape so that integration is a non-event:

```python
# registry/resolver.py
@dataclass
class ResolvedCitation:
    raw: str
    target_type: str        # "section" | "statute"
    target_id: str
    confidence: float

def resolve_citation(raw_text: str) -> ResolvedCitation | None:
    """Normalise a raw citation string (as emitted by core/ner.py's STATUTE/PROVISION
    entities) to a registry target. Returns None if it can't be resolved with
    confidence -- never guess. A None here is what feeds the coverage/unresolved
    count in your reporting."""
```

```python
# verify/registry_check.py
@dataclass
class RegistryResult:
    status: str              # "in_force" | "repealed" | "superseded" | "not_found"
    as_of: date
    note: str | None = None  # e.g. "repealed 2024-07-01; see BNS §318"

def check_citation(citation: ResolvedCitation, as_of: date) -> RegistryResult:
    """Never raises. 'not_found' is a valid, honest return -- per CLAUDE.md's I2,
    a safety component never fails open or substitutes a guessed default. gate.py
    will treat 'not_found' the same as any other non-'in_force' status: it strikes
    the claim, it does not pass it through."""
```

This mirrors **I2** directly: no fabricated status, ever. If you're unsure whether a
citation is in force, that's `"not_found"`, not a guess dressed up as an answer.

---

## 7. Git workflow

```bash
git clone https://github.com/kramjiy/smartlawai.git
cd smartlawai
git checkout -b m4-registry
```

Work on that branch, push it, open a PR against `main` rather than pushing straight
to `main` — with two people now, that's worth the small extra step. Since
`registry/`, `verify/registry_check.py`, and `tests/test_registry.py` are all new
paths nobody else is touching, merge conflicts should be close to zero.

Pull `main` at the start of every session (`git pull origin main`) — the other
person's commits land there independently, and you want the latest `CLAUDE.md`/
`PLAN.md` if either gets updated.

Two automated backups already mirror `main` on a delay (`.github/workflows/`) — not
something you need to do anything about, just know your work is protected once it's
merged.

---

## 8. Setup

```bash
git clone https://github.com/kramjiy/smartlawai.git
cd smartlawai
git checkout -b m4-registry
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # duckdb + pytest, everything M4 needs
pytest -q                        # should already be green: existing tests for
                                  # local backend, API, metrics, NER
```

---

## 9. Concrete first steps, in order

1. Track down the MHA IPC↔BNS/CrPC↔BNSS/IEA↔BSA comparison tables. Check
   machine-readable vs. scanned **immediately** — this is the single biggest
   unknown in your whole estimate and determines the rest of the week.
2. Scaffold `registry/` — schema (§2), following `adapters/local.py`'s DuckDB style.
3. Run `core/ner.py`'s extraction over whatever corpus is available; `Counter()` the
   results to find the ~95%-coverage citation set.
4. Parse the MHA tables into `sections`/`statutes` rows; hand-verify a stratified
   100-entry sample against the Gazette; record the verification accuracy.
5. Build `registry/resolver.py` and `verify/registry_check.py` to the exact contract
   in §6.
6. Write and pass `test_repealed_ipc_section_is_struck`.
7. Arrange the independent 200-entry audit — by construction this can't be you (see
   threat T10 in `RESEARCH.md` §8); this is a people/scheduling question to raise,
   not something to solve solo.

---

## 10. If something feels ambiguous

Don't guess past these — they affect the paper's argument, not just code:
- MHA tables turn out scanned/unreliable
- The RR↔SUMM section-pair decision (M1, owned by the other track) affects what
  corpus you're extracting citations from
- Any point where "demand-driven ~95% coverage" starts looking like it's covering
  meaningfully less in practice — report the real number, don't quietly lower the bar
