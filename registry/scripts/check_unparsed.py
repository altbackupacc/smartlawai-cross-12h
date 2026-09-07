"""Correct ground-truth check + triage of rows the parser could not interpret."""

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.ingest.mha_tables import (
    _CHAPTER_ROW,
    _NEW_MARKERS,
    DATA,
    TABLES,
    _section_numbers,
    _TableParser,
    parse_all,
)

rows, _ = parse_all()
by_old = {}
for r in rows:
    for o in r.old_sections:
        by_old.setdefault((r.old_act, o), []).append(r)

# Lettered sections matter: IPC 124 and 124A are different offences that map to
# different BNS sections. The earlier check conflated them.
EXPECT = {
    ("ipc-1860", "420"): "318",    # cheating -- the done-when test
    ("ipc-1860", "302"): "103",    # murder
    ("ipc-1860", "376"): "64",     # rape
    ("ipc-1860", "124"): "151",    # assaulting President/Governor
    ("ipc-1860", "124A"): None,    # sedition: DELETED, no BNS successor
    ("ipc-1860", "120"): "60",     # concealing design
    ("ipc-1860", "120B"): "61",    # criminal conspiracy (punishment)
    ("ipc-1860", "499"): "356",    # defamation
    ("ipc-1860", "304A"): "106",   # death by negligence
    ("ipc-1860", "498A"): "85",    # cruelty by husband/relatives
    ("crpc-1973", "154"): "173",   # FIR
    ("crpc-1973", "161"): "180",   # police statements
    ("iea-1872", "65B"): "63",     # electronic records
    ("iea-1872", "32"): "26",      # dying declaration
}

print("=== ground-truth spot checks (corrected) ===")
ok = fail = 0
for (act, old), want in sorted(EXPECT.items()):
    hits = by_old.get((act, old), [])
    got = sorted({n for h in hits for n in h.new_sections})
    if want is None:
        hit = bool(hits) and all(h.relation == "repealed_no_successor" for h in hits)
        print("  %s %-10s %-5s -> expected REPEALED/no successor, got %s"
              % ("OK  " if hit else "MISS", act, old,
                 sorted({h.relation for h in hits}) or "[] (row missing)"))
        ok, fail = ok + hit, fail + (not hit)
        continue
    hit = want in got
    ok, fail = ok + hit, fail + (not hit)
    print("  %s %-10s %-5s -> expected %-4s got %s"
          % ("OK  " if hit else "MISS", act, old, want, got or "[]"))
print("\n  %d/%d correct" % (ok, ok + fail))

# ------------------------------------------------------------------ #
print("\n=== unparsed row triage ===")
for fname in TABLES:
    p = _TableParser()
    p.feed((DATA / fname).read_text(encoding="utf-8", errors="replace"))
    unparsed = []
    for r in p.rows:
        if len(r) != 2:
            unparsed.append(r)
            continue
        left, right = r[0].strip(), r[1].strip()
        if ("sanhita" in left.lower() or "adhiniyam" in left.lower()) and (
            "penal code" in right.lower() or "criminal procedure" in right.lower()
            or "evidence act" in right.lower()):
            continue
        if _CHAPTER_ROW.match(left) or _CHAPTER_ROW.match(right):
            continue
        ns, os_ = _section_numbers(left), _section_numbers(right)
        if _NEW_MARKERS.search(right) and not os_ and ns:
            continue
        if os_ and not ns:
            continue
        if ns and os_:
            continue
        unparsed.append([left, right])

    print("\n  --- %s : %d unparsed ---" % (fname, len(unparsed)))
    kinds = Counter()
    for left, right in unparsed:
        if not left and not right:
            kinds["both empty"] += 1
        elif not right:
            kinds["right empty"] += 1
        elif not left:
            kinds["left empty"] += 1
        else:
            kinds["no number found either side"] += 1
    for k, v in kinds.most_common():
        print("      %-32s %d" % (k, v))
    for left, right in unparsed[:6]:
        print("      L=%-46s R=%s" % (left[:46], right[:46]))
