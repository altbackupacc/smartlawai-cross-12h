"""Cross-validate correspondence-table sections against India Code's authoritative lists.

For BNS/BNSS/BSA we now have two independent sources: the MHA correspondence
tables (parsed) and India Code's section index (authoritative). Any section the
parser produced that India Code does not list is a parse error, not a section.

This is the check that turns "parsed" into "corroborated", and it is the same
machinery that will validate IPC/CrPC/IEA once a list for them is sourced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.ingest.indiacode import fetch_sections
from registry.schema import RegistryDB

ACT_IDS = json.loads((REPO / "registry" / "out" / "act_ids.json").read_text(encoding="utf-8"))
CHECKABLE = ["bns-2023", "bnss-2023", "bsa-2023"]

db = RegistryDB()
problems = {}

for act in CHECKABLE:
    official = {s["number"] for s in fetch_sections(ACT_IDS[act]["act_id"])}
    ours = {r[0] for r in db.db.execute(
        "SELECT number FROM SECTIONS WHERE statute_id = ?", [act]).fetchall()}

    spurious = sorted(ours - official)
    missing = sorted(official - ours)
    print(f"=== {act} ===")
    print("  india code : %d sections" % len(official))
    print("  registry   : %d sections" % len(ours))
    print("  spurious   : %d %s" % (len(spurious), spurious[:20]))
    print("  missing    : %d %s" % (len(missing), missing[:20]))
    problems[act] = {"spurious": spurious, "missing": missing}

    for n in spurious:
        row = db.db.execute(
            "SELECT id, source_doc FROM SECTIONS WHERE statute_id=? AND number=?",
            [act, n]).fetchone()
        sup = db.db.execute(
            "SELECT from_section_id, to_section_id FROM SUPERSESSION WHERE to_section_id=?",
            [f"{act}-s{n}"]).fetchall()
        print("     spurious %-14s from=%s  supersession rows=%s" % (n, row[1], sup[:3]))
    print()

(REPO / "registry" / "out" / "section_validation.json").write_text(
    json.dumps(problems, indent=2), encoding="utf-8")
print("wrote M4/out/section_validation.json")
db.close()
