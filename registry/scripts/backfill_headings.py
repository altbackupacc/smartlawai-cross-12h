"""Backfill IPC / CrPC / IEA section headings from the correspondence tables.

India Code does not index the three repealed codes, so their headings have no
authoritative source. The correspondence tables' old-code column carries them,
and that is the only machine-readable source available. Headings written here
are marked `source_doc = 'ncrb:<file>#old-column'` so they are distinguishable
from India Code headings during the verification pass -- they are weaker
evidence and should be sampled more heavily.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.ingest.mha_tables import (
    _CHAPTER_ROW,
    DATA,
    TABLES,
    _TableParser,
    section_headings,
)
from registry.schema import RegistryDB


def main() -> int:
    # act -> {section number -> heading}
    found: dict[str, dict[str, str]] = defaultdict(dict)
    conflicts: dict[str, list] = defaultdict(list)

    for fname, (_new_act, old_act) in TABLES.items():
        p = _TableParser()
        p.feed((DATA / fname).read_text(encoding="utf-8", errors="replace"))
        for r in p.rows:
            if len(r) != 2:
                continue
            left, right = r[0].strip(), r[1].strip()
            if _CHAPTER_ROW.match(left) or _CHAPTER_ROW.match(right):
                continue
            for num, head in section_headings(right).items():
                prev = found[old_act].get(num)
                if prev and prev != head:
                    # Keep the longer rendering; record that they differed.
                    conflicts[old_act].append((num, prev, head))
                    if len(head) <= len(prev):
                        continue
                found[old_act][num] = head

    db = RegistryDB()
    from datetime import date
    COMMENCEMENT = date(2024, 7, 1)

    print("=== headings recovered from the old-code column ===")
    for act in ("ipc-1860", "crpc-1973", "iea-1872"):
        heads = found.get(act, {})
        before = db.db.execute(
            "SELECT count(*) FROM SECTIONS WHERE statute_id = ?", [act]).fetchone()[0]
        existing = {r[0] for r in db.db.execute(
            "SELECT number FROM SECTIONS WHERE statute_id = ?", [act]).fetchall()}

        # Sections that appear in the tables but were never created, because the
        # row they sit in did not parse as a *mapping*. They are still real
        # provisions of the repealed code -- IPC 337/338, 416, 431-436, 450/451
        # among them -- and leaving them out makes the registry answer
        # "not_found" for citations that genuinely existed.
        added = 0
        for num, head in heads.items():
            if num in existing:
                continue
            db.add_section(
                id=f"{act}-s{num}", statute_id=act, kind="s", number=num,
                heading=head, in_force_to=COMMENCEMENT,
                source_doc="ncrb-old-column", verified_by=None,
            )
            added += 1

        for num, head in heads.items():
            db.db.execute(
                "UPDATE SECTIONS SET heading = ?, "
                "source_doc = COALESCE(source_doc, '') || ' | ncrb-old-column' "
                "WHERE id = ? AND heading IS NULL",
                [head, f"{act}-s{num}"])

        after = db.db.execute(
            "SELECT count(*) FROM SECTIONS WHERE statute_id = ?", [act]).fetchone()[0]
        filled = db.db.execute(
            "SELECT count(*) FROM SECTIONS WHERE statute_id = ? AND heading IS NOT NULL",
            [act]).fetchone()[0]
        print("  %-12s %4d -> %4d sections (+%d recovered) | %4d headings | %4d filled (%.0f%%)"
              % (act, before, after, added, len(heads), filled, 100 * filled / max(after, 1)))
        if conflicts.get(act):
            print("       %d rows gave two renderings of the same section (longer kept)"
                  % len(conflicts[act]))

    print("\n=== sample ===")
    for row in db.db.execute("""
        SELECT id, heading FROM SECTIONS
        WHERE statute_id='ipc-1860' AND heading IS NOT NULL
          AND number IN ('302','420','124A','498A','304A','376','120B')
        ORDER BY id""").fetchall():
        print("  %-18s %s" % (row[0], (row[1] or "")[:72]))

    total_h, total_s = db.db.execute(
        "SELECT sum(CASE WHEN heading IS NOT NULL THEN 1 ELSE 0 END), count(*) FROM SECTIONS"
    ).fetchone()
    print("\nregistry-wide heading coverage: %d / %d (%.1f%%)"
          % (total_h, total_s, 100 * total_h / total_s))
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
