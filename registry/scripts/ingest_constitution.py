"""Load the Constitution's articles into the registry.

Articles are stored with `kind='art'` so they never collide with sections
(`constitution-1950-art32` vs a hypothetical section 32).

Omitted articles are recorded with `status='omitted'` and a NULL `in_force_to`.
We know *that* they were removed -- the official arrangement of articles brackets
them as "[31. Omitted.]" -- but not *when*, because the amendment date sits in
body footnotes rather than the arrangement. Writing a guessed date would
fabricate authority (CLAUDE.md I2); leaving the status unset would report a
repealed article as good law. Recording the fact without the date is the only
honest option available from this source.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.ingest.constitution import parse_articles
from registry.schema import RegistryDB

COMMENCEMENT = date(1950, 1, 26)
SRC = "https://www.legislative.gov.in/"
SRC_DOC = "legislative.gov.in:COI_english_2025-11.pdf"


def main() -> int:
    arts, stats = parse_articles()
    print(f"parse stats: {stats}")
    print("articles parsed: %d" % len(arts))

    db = RegistryDB()
    before = db.counts()

    n_dated = n_undated = 0
    for num, a in arts.items():
        sid = f"constitution-1950-art{num}"
        # An omission with a known effective date becomes a real in-force
        # interval. Only omissions whose date this source does not state fall
        # back to the `status` marker.
        db.add_section(
            id=sid,
            statute_id="constitution-1950",
            kind="art",
            number=num,
            heading=a.heading,
            in_force_from=COMMENCEMENT,
            in_force_to=(a.omitted_on if a.omitted and a.omitted_on else None),
            status=("omitted" if a.omitted and not a.omitted_on else None),
            source_url=SRC,
            source_doc=SRC_DOC,
            verified_by=None,
        )
        if not a.omitted:
            continue
        if a.omitted_on:
            n_dated += 1
            db.add_supersession(
                id=f"sup-coi-{num}",
                from_section_id=sid,
                to_section_id=None,
                from_statute_id="constitution-1950",
                to_statute_id=None,
                effective_on=a.omitted_on,
                relation="repealed_no_successor",
                note="Omitted by %s" % (a.omitted_by or "a Constitution Amendment Act"),
                source_doc=SRC_DOC,
            )
        else:
            n_undated += 1

    after = db.counts()
    print("\nSECTIONS %d -> %d (+%d)" % (before["SECTIONS"], after["SECTIONS"],
                                        after["SECTIONS"] - before["SECTIONS"]))

    n_om = db.db.execute(
        "SELECT count(*) FROM SECTIONS WHERE statute_id='constitution-1950' "
        "AND status='omitted'").fetchone()[0]
    n_all = db.db.execute(
        "SELECT count(*) FROM SECTIONS WHERE statute_id='constitution-1950'").fetchone()[0]
    print("constitution articles: %d (%d marked omitted)" % (n_all, n_om))

    print("\n--- omitted articles (the second staleness instance) ---")
    for row in db.db.execute(
        "SELECT number FROM SECTIONS WHERE statute_id='constitution-1950' "
        "AND status='omitted' ORDER BY CAST(regexp_extract(number,'^\\d+') AS INT), number"
    ).fetchall():
        print(f"   art {row[0]}", end="")
    print()

    print("\n--- spot check ---")
    for n in ("14", "21", "31", "32", "51A", "226", "370"):
        row = db.db.execute(
            "SELECT number, status, heading FROM SECTIONS WHERE id=?",
            [f"constitution-1950-art{n}"]).fetchone()
        print("   art %-5s status=%-8s %s" % (row[0], row[1] or "-", (row[2] or "")[:56]))

    tot_h, tot_s = db.db.execute(
        "SELECT sum(CASE WHEN heading IS NOT NULL THEN 1 ELSE 0 END), count(*) FROM SECTIONS"
    ).fetchone()
    print("\nregistry: %d sections, %d with headings (%.1f%%)"
          % (tot_s, tot_h, 100 * tot_h / tot_s))
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
