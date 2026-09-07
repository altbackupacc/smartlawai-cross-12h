"""Seed the STATUTES table with the 10 seed statutes (PLAN.md M4).

Commencement and repeal dates below are the well-established public record.
They are seeded with `verified_by` left NULL on purpose: PLAN.md requires
primary sourcing (Gazette of India), and until a human has checked each date
against the Gazette these rows are *unverified*, and the registry says so.
Threat T10 is precisely the risk of asserting authority we have not checked.

The three repeal rows (IPC/CrPC/IEA -> BNS/BNSS/BSA, 2024-07-01) are the
centrepiece of the natural experiment in RESEARCH.md 2.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.schema import RegistryDB

INDIA_CODE = "https://indiacode.gov.in/"  # migrated from indiacode.nic.in (2026)

# id, short_title, long_title, year, in_force_from, repealed_on, repealed_by
SEED = [
    ("ipc-1860", "Indian Penal Code", "The Indian Penal Code, 1860 (Act 45 of 1860)",
     1860, date(1862, 1, 1), date(2024, 7, 1), "bns-2023"),
    ("crpc-1973", "Code of Criminal Procedure",
     "The Code of Criminal Procedure, 1973 (Act 2 of 1974)",
     1973, date(1974, 4, 1), date(2024, 7, 1), "bnss-2023"),
    ("iea-1872", "Indian Evidence Act", "The Indian Evidence Act, 1872 (Act 1 of 1872)",
     1872, date(1872, 9, 1), date(2024, 7, 1), "bsa-2023"),
    ("bns-2023", "Bharatiya Nyaya Sanhita",
     "The Bharatiya Nyaya Sanhita, 2023 (Act 45 of 2023)",
     2023, date(2024, 7, 1), None, None),
    ("bnss-2023", "Bharatiya Nagarik Suraksha Sanhita",
     "The Bharatiya Nagarik Suraksha Sanhita, 2023 (Act 46 of 2023)",
     2023, date(2024, 7, 1), None, None),
    ("bsa-2023", "Bharatiya Sakshya Adhiniyam",
     "The Bharatiya Sakshya Adhiniyam, 2023 (Act 47 of 2023)",
     2023, date(2024, 7, 1), None, None),
    ("contract-1872", "Indian Contract Act", "The Indian Contract Act, 1872 (Act 9 of 1872)",
     1872, date(1872, 9, 1), None, None),
    ("arbitration-1996", "Arbitration and Conciliation Act",
     "The Arbitration and Conciliation Act, 1996 (Act 26 of 1996)",
     1996, date(1996, 8, 22), None, None),
    ("consumer-2019", "Consumer Protection Act",
     "The Consumer Protection Act, 2019 (Act 35 of 2019)",
     2019, date(2020, 7, 20), None, None),
    ("cpc-1908", "Code of Civil Procedure", "The Code of Civil Procedure, 1908 (Act 5 of 1908)",
     1908, date(1909, 1, 1), None, None),
    ("ni-1881", "Negotiable Instruments Act",
     "The Negotiable Instruments Act, 1881 (Act 26 of 1881)",
     1881, date(1882, 3, 1), None, None),
    ("it-2000", "Information Technology Act",
     "The Information Technology Act, 2000 (Act 21 of 2000)",
     2000, date(2000, 10, 17), None, None),
    ("stamp-1899", "Indian Stamp Act", "The Indian Stamp Act, 1899 (Act 2 of 1899)",
     1899, date(1899, 7, 1), None, None),
    ("constitution-1950", "Constitution of India", "The Constitution of India",
     1950, date(1950, 1, 26), None, None),
]


def main() -> int:
    db = RegistryDB()
    for (sid, short, long, year, iff, rep, repby) in SEED:
        db.add_statute(
            id=sid, short_title=short, long_title=long, year=year,
            jurisdiction="IN", in_force_from=iff, repealed_on=rep,
            repealed_by_statute_id=repby, source_url=INDIA_CODE,
        )
    print("counts:", db.counts())
    print("\nrepealed statutes (the natural experiment):")
    for row in db.db.execute(
        "SELECT id, short_title, repealed_on, repealed_by_statute_id "
        "FROM STATUTES WHERE repealed_on IS NOT NULL ORDER BY id"
    ).fetchall():
        print("  %-12s %-34s repealed %s -> %s" % row)
    print("\nverification status: 0 of %d statutes verified against the Gazette "
          "(verified_by is NULL on every row)" % len(SEED))
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
