"""Load parsed correspondence rows into SECTIONS and SUPERSESSION.

Every row records `source_doc` and leaves `verified_by` NULL: parsed is not
verified. PLAN.md M4 requires a hand-verified stratified sample of 100 entries
plus an independent 200-entry audit before any of this counts as checked.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.ingest.mha_tables import COMMENCEMENT, TABLES, parse_all
from registry.schema import RegistryDB

SRC = {v[0]: k for k, v in TABLES.items()}  # new_act -> source filename


def main() -> int:
    rows, stats = parse_all()
    db = RegistryDB()

    # --- sections: every section named on either side of the tables ---
    seen: set[tuple[str, str]] = set()
    for r in rows:
        for act, nums in ((r.new_act, r.new_sections), (r.old_act, r.old_sections)):
            for n in nums:
                if (act, n) in seen:
                    continue
                seen.add((act, n))
                repealed = act in ("ipc-1860", "crpc-1973", "iea-1872")
                db.add_section(
                    id=f"{act}-s{n}",
                    statute_id=act,
                    kind="s",
                    number=n,
                    in_force_to=(COMMENCEMENT if repealed else None),
                    source_doc=SRC.get(r.new_act),
                    verified_by=None,
                )

    # --- supersession: the many-to-many record ---
    # Deduplicated on (from, to, relation). The tables map at sub-section
    # granularity and we fold to the parent section, so CrPC 46(1)..46(5) all
    # yield the same CrPC 46 -> BNSS 43 mapping. Storing it five times inflates
    # the supersession count and makes that mapping five times likelier to be
    # drawn in a verification sample -- a silent sampling bias.
    seen_sup: set[tuple[str, str | None, str]] = set()
    n_sup = 0
    n_dupe = 0
    for i, r in enumerate(rows):
        if r.relation == "new_provision":
            continue
        src = SRC.get(r.new_act)
        if r.relation == "repealed_no_successor":
            for o in r.old_sections:
                key = (f"{r.old_act}-s{o}", None, r.relation)
                if key in seen_sup:
                    n_dupe += 1
                    continue
                seen_sup.add(key)
                db.add_supersession(
                    id="sup-%06d-%s" % (i, o),
                    from_section_id=f"{r.old_act}-s{o}",
                    to_section_id=None,
                    from_statute_id=r.old_act,
                    to_statute_id=None,
                    effective_on=COMMENCEMENT,
                    relation="repealed_no_successor",
                    note=f"repealed {COMMENCEMENT}; no corresponding provision",
                    source_doc=src,
                )
                n_sup += 1
            continue
        for o in r.old_sections:
            for nsec in r.new_sections:
                key = (f"{r.old_act}-s{o}", f"{r.new_act}-s{nsec}", r.relation)
                if key in seen_sup:
                    n_dupe += 1
                    continue
                seen_sup.add(key)
                db.add_supersession(
                    id="sup-%06d-%s-%s" % (i, o, nsec),
                    from_section_id=f"{r.old_act}-s{o}",
                    to_section_id=f"{r.new_act}-s{nsec}",
                    from_statute_id=r.old_act,
                    to_statute_id=r.new_act,
                    effective_on=COMMENCEMENT,
                    relation="maps_to",
                    source_doc=src,
                )
                n_sup += 1

    # --- denormalise the unambiguous 1:1 cases onto SECTIONS ---
    db.db.execute("""
        UPDATE SECTIONS SET superseded_by_section_id = m.to_section_id
        FROM (SELECT from_section_id, min(to_section_id) AS to_section_id
              FROM SUPERSESSION WHERE relation = 'maps_to' AND to_section_id IS NOT NULL
              GROUP BY from_section_id HAVING count(DISTINCT to_section_id) = 1) AS m
        WHERE SECTIONS.id = m.from_section_id""")

    print("parse stats:")
    for f, s in stats.items():
        print("  %-16s %s" % (f, s))
    print("\nsupersession rows written: %d" % n_sup)
    print(f"registry counts: {db.counts()}")

    one_to_one = db.db.execute(
        "SELECT count(*) FROM SECTIONS WHERE superseded_by_section_id IS NOT NULL"
    ).fetchone()[0]
    print("sections with an unambiguous 1:1 successor: %d" % one_to_one)

    print("\n--- IPC 420, the done-when test's subject ---")
    for row in db.db.execute("""
        SELECT s.id, s.in_force_to, sup.to_section_id, sup.relation
        FROM SECTIONS s LEFT JOIN SUPERSESSION sup ON sup.from_section_id = s.id
        WHERE s.id = 'ipc-1860-s420'""").fetchall():
        print("   {}  in_force_to={}  ->  {} ({})".format(*row))

    print("\n--- IPC 124A (sedition): repealed with no successor ---")
    for row in db.db.execute("""
        SELECT s.id, s.in_force_to, sup.to_section_id, sup.relation
        FROM SECTIONS s LEFT JOIN SUPERSESSION sup ON sup.from_section_id = s.id
        WHERE s.id = 'ipc-1860-s124A'""").fetchall():
        print("   {}  in_force_to={}  ->  {} ({})".format(*row))

    print("\nverification: 0 rows verified (verified_by IS NULL on all %d sections)"
          % db.counts()["SECTIONS"])
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
