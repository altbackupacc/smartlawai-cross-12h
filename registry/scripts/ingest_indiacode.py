"""Pull every section of every resolved statute from India Code into the registry."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.ingest.indiacode import fetch_sections
from registry.schema import RegistryDB

ACT_IDS = REPO / "registry" / "out" / "act_ids.json"
SRC = "https://indiacode.gov.in/"


def _parse_enf(v: str | None) -> date | None:
    """India Code writes enforcement dates several ways: '1-7-2024',
    '01-07-2024', '1872-09-01', or a paragraph of notification prose."""
    if not v:
        return None
    v = v.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            from datetime import datetime
            return datetime.strptime(v[:10], fmt).date()
        except ValueError:
            continue
    return None


def main() -> int:
    if not ACT_IDS.exists():
        print("run find_act_ids.py first", file=sys.stderr)
        return 1
    acts = json.loads(ACT_IDS.read_text(encoding="utf-8"))
    db = RegistryDB()

    before = db.counts()
    print(f"before: {before}\n")
    total_new = 0
    report = {}

    for reg_id, info in acts.items():
        try:
            secs = fetch_sections(info["act_id"])
        except Exception as e:  # noqa: BLE001
            print("  %-18s FAILED: %s" % (reg_id, e))
            report[reg_id] = {"error": str(e)}
            continue

        enf = _parse_enf(info.get("enforcement_date"))
        n_rep = sum(1 for s in secs if s["repealed"])
        n_text = sum(1 for s in secs if s.get("text"))
        for s in secs:
            db.add_section(
                id="{}-s{}".format(reg_id, s["number"]),
                statute_id=reg_id,
                kind="s",
                number=s["number"],
                heading=s["heading"],
                text=s.get("text"),
                in_force_from=enf,
                # A section flagged repealed in India Code has been repealed
                # individually; we do not know the date from this feed, so the
                # date stays NULL and only the flag is recorded via in_force_to
                # being left for the statute-level inheritance to handle.
                in_force_to=None,
                source_url=SRC,
                source_doc="indiacode:{}".format(info["act_id"]),
                verified_by=None,
            )
        total_new += len(secs)
        report[reg_id] = {
            "act_id": info["act_id"], "sections": len(secs),
            "repealed_sections": n_rep, "enforcement_date": str(enf),
            "sections_with_text": n_text,
        }
        print("  %-18s %4d sections  %4d with text  enf=%s"
              % (reg_id, len(secs), n_text, enf))

    after = db.counts()
    print(f"\nafter: {after}")
    print("sections ingested this run: %d" % total_new)

    (REPO / "registry" / "out" / "indiacode_ingest_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- sections per statute now in registry ---")
    for row in db.db.execute(
        "SELECT statute_id, count(*) AS n, "
        "       sum(CASE WHEN heading IS NOT NULL THEN 1 ELSE 0 END) AS with_heading "
        "FROM SECTIONS GROUP BY statute_id ORDER BY n DESC"
    ).fetchall():
        print("  %-20s %5d sections  %5d with headings" % row)
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
