"""Run the correspondence-table parser and check it against known ground truth."""

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.ingest.mha_tables import parse_all

rows, stats = parse_all()

print("=== parse stats ===")
for f, s in stats.items():
    print(f"  {f}")
    print(f"     {s}")

print("\ntotal mapping rows: %d" % len(rows))
print(f"relations: {dict(Counter(r.relation for r in rows))}")

# --- ground truth spot checks -------------------------------------------
# Publicly documented IPC -> BNS correspondences. NOTE the lettered sections:
# IPC 124 (assaulting the President) and 124A (sedition) are different
# offences with different fates, as are 120 and 120B. Conflating them produces
# false failures -- the fuller, corrected set lives in check_unparsed.py.
EXPECT = {
    ("ipc-1860", "420"): "318",   # cheating -> BNS 318 (the done-when test)
    ("ipc-1860", "302"): "103",   # murder
    ("ipc-1860", "376"): "64",    # rape
    ("ipc-1860", "124"): "151",   # assaulting President/Governor
    ("ipc-1860", "499"): "356",   # defamation
    ("ipc-1860", "120"): "60",    # concealing design (120B -> 61 is conspiracy)
}

print("\n=== ground-truth spot checks (IPC -> BNS) ===")
by_old = {}
for r in rows:
    for o in r.old_sections:
        by_old.setdefault((r.old_act, o), []).append(r)

ok = fail = 0
for (act, old), want in EXPECT.items():
    hits = by_old.get((act, old), [])
    got = sorted({n for h in hits for n in h.new_sections})
    mark = "OK " if want in got else "MISS"
    if want in got:
        ok += 1
    else:
        fail += 1
    print("  %s IPC %-4s -> expected BNS %-4s got %s" % (mark, old, want, got or "[]"))
print("\n  %d/%d ground-truth mappings correct" % (ok, ok + fail))

print("\n=== sample parsed rows ===")
for r in rows[:8]:
    print("  %-9s %s %s <- %s %s" % (r.relation, r.new_act, r.new_sections,
                                     r.old_act, r.old_sections))

print("\n=== one-to-many / many-to-one (why SUPERSESSION is a table) ===")
multi = [r for r in rows if len(r.old_sections) > 1 or len(r.new_sections) > 1]
print("  %d rows are not 1:1 (%.1f%% of mapped rows)"
      % (len(multi), 100 * len(multi) / max(len(rows), 1)))
for r in multi[:6]:
    print(f"     {r.new_act} {r.new_sections} <- {r.old_act} {r.old_sections}")
