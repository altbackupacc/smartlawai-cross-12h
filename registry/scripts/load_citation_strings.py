"""Populate CITATION_STRINGS — the third table in PLAN.md M4's schema.

    citation_strings(id, raw, normalised, target_type, target_id, confidence)

It is the observed-citation layer: every distinct raw citation string the corpus
actually contains, with what it normalises to and how confident that mapping is.
It serves three purposes:

1. **A resolver cache** — `resolve_citation` can short-circuit on a known raw
   string instead of re-parsing.
2. **The unresolved register** — strings that resolve to nothing are stored with
   `target_id = NULL`, so "flagged, never silently dropped" (PLAN.md M4) is a
   queryable fact rather than a claim.
3. **Audit sampling frame** — the verification pass needs to sample citations
   weighted by real-world frequency, not uniformly over the registry.

Confidence reflects how the citation was attributed during extraction, not how
sure we are that the section exists (that is settled by whether it resolves):

    statute_span       1.00  the act name sat inside the citation itself
    nearest_preceding  0.90 / 0.60  act inferred from context, near / far
    article_default    0.80  bare "Article N" taken as the Constitution
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.schema import RegistryDB
from registry.types import TARGET_SECTION, TARGET_STATUTE

NEAR_WINDOW = 600

CONF = {
    "statute_span": 1.0,
    "article_default": 0.8,
    "act_only": 0.9,
}


def confidence(row) -> float:
    if row.attribution == "nearest_preceding":
        d = row.distance
        return 0.9 if (d is not None and d <= NEAR_WINDOW) else 0.6
    return CONF.get(row.attribution, 0.5)


def main() -> int:
    src = REPO / "registry" / "out" / "citations.parquet"
    if not src.exists():
        print("run extract_citations.py first", file=sys.stderr)
        return 1

    # Orphans are KEPT. They are the unresolved register: citations to acts
    # outside the seed set, stored with a NULL target so that "flagged, never
    # silently dropped" (PLAN.md M4) is a queryable fact rather than a claim.
    # Filtering them out here would make the table report 100% resolution by
    # construction, which is precisely the dishonest number this project exists
    # to avoid.
    df = pd.read_parquet(src)
    df = df[df.raw.notna()]

    db = RegistryDB()
    known = {r[0] for r in db.db.execute("SELECT id FROM SECTIONS").fetchall()}
    known_acts = {r[0] for r in db.db.execute("SELECT id FROM STATUTES").fetchall()}

    # A bare act mention ("Indian Penal Code") has no section_id but is still a
    # perfectly good *statute* target -- the contract in `types.py` has both
    # target types. Grouping on section_id alone counted 1,855 of these as
    # unresolved, which understated coverage and misrepresented the tail.
    df = df.assign(conf=df.apply(confidence, axis=1))
    grp = (df.groupby(["raw", "section_id", "act_id"], dropna=False)
             .agg(occurrences=("raw", "size"), confidence=("conf", "max"))
             .reset_index())

    n_sec = n_act = n_unres = 0
    rows = []
    for r in grp.itertuples(index=False):
        sid = r.section_id if isinstance(r.section_id, str) else None
        aid = r.act_id if isinstance(r.act_id, str) else None

        if sid and sid in known:
            ttype, tid, norm = TARGET_SECTION, sid, sid
            n_sec += 1
        elif aid and aid in known_acts:
            ttype, tid, norm = TARGET_STATUTE, aid, aid
            n_act += 1
        else:
            ttype, tid, norm = None, None, (sid or aid)
            n_unres += 1

        rows.append((
            "cs-" + hashlib.sha1(
                ("{}|{}".format(r.raw, tid or "")).encode("utf-8")).hexdigest()[:16],
            r.raw,
            norm,
            ttype,
            tid,                                   # never point at a missing row
            float(r.confidence),
            int(r.occurrences),
        ))

    db.db.execute("DELETE FROM CITATION_STRINGS")
    db.db.executemany(
        "INSERT INTO CITATION_STRINGS "
        "(id, raw, normalised, target_type, target_id, confidence, occurrences) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    print("distinct (raw, target) pairs : %d" % len(rows))
    print("  resolved to a section       : %d" % n_sec)
    print("  resolved to a statute       : %d" % n_act)
    print("  unresolved (kept, flagged)  : %d" % n_unres)
    print(f"counts: {db.counts()}")

    print("\n--- most frequent resolved citations ---")
    for row in db.db.execute(
        "SELECT raw, target_id, occurrences, confidence FROM CITATION_STRINGS "
        "WHERE target_id IS NOT NULL ORDER BY occurrences DESC LIMIT 8").fetchall():
        print("  %-38s -> %-28s x%-5d conf=%.2f" % row)

    print("\n--- most frequent UNRESOLVED (the honest tail) ---")
    for row in db.db.execute(
        "SELECT raw, occurrences FROM CITATION_STRINGS "
        "WHERE target_id IS NULL ORDER BY occurrences DESC LIMIT 8").fetchall():
        print("  %-38s x%d" % row)

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
