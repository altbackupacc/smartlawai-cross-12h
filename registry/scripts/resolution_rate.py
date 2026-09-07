"""What fraction of real corpus citations does the registry actually resolve?

This is the coverage number PLAN.md M4 asks to be reported. Unlike the earlier
frequency analysis, every citation here is checked against the registry, so
invented section numbers are rejected rather than counted.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.resolver import _db
from registry.types import TARGET_SECTION, ResolvedCitation
from verify.registry_check import check_citation

AS_OF = date(2026, 9, 5)

df = pd.read_parquet(REPO / "registry" / "out" / "citations.parquet")
conn = _db()

known = {r[0] for r in conn.execute("SELECT id FROM SECTIONS").fetchall()}

# Every row that named a section at all (attributed by any method).
sec_rows = df[df.section_id.notna()]
total = len(sec_rows)
counts = Counter(sec_rows.section_id)

resolved = sum(c for s, c in counts.items() if s in known)
unresolved = total - resolved

print("=== registry resolution over the corpus ===")
print("  section-level citation instances : %d" % total)
print("  resolved against registry        : %d (%.1f%%)" % (resolved, 100 * resolved / total))
print("  unresolved (flagged, not dropped): %d (%.1f%%)" % (unresolved, 100 * unresolved / total))
print("  distinct sections cited          : %d" % len(counts))
print("  distinct resolved                : %d" % sum(1 for s in counts if s in known))

print("\n=== per-act resolution ===")
rows = []
for act in sorted({s.rsplit("-s", 1)[0].rsplit("-art", 1)[0] for s in counts}):
    tot = sum(c for s, c in counts.items() if s.startswith(act + "-"))
    ok = sum(c for s, c in counts.items() if s.startswith(act + "-") and s in known)
    if tot:
        rows.append((act, tot, ok, 100 * ok / tot))
for act, tot, ok, pct in sorted(rows, key=lambda r: -r[1]):
    print("  %-22s %6d cited  %6d resolved  %5.1f%%" % (act, tot, ok, pct))

print(f"\n=== authority status of resolved citations, as of {AS_OF} ===")
status = Counter()
for sid, c in counts.items():
    if sid not in known:
        continue
    res = check_citation(
        ResolvedCitation(raw=sid, target_type=TARGET_SECTION, target_id=sid, confidence=1.0),
        as_of=AS_OF, db=conn)
    status[res.status] += c
tot_res = sum(status.values())
for k, v in status.most_common():
    print("  %-12s %7d (%.1f%%)" % (k, v, 100 * v / tot_res))

stale = tot_res - status.get("in_force", 0)
print("\n  >>> %d of %d resolved citations (%.1f%%) are NOT in force today."
      % (stale, tot_res, 100 * stale / tot_res))
print("      Entailment verification cannot detect any of them.")
