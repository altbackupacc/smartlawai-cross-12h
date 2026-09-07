"""How large must the registry be to cover what the corpus actually cites?

Answers the question PLAN.md M4 poses ("report coverage as a number") and sizes
the exhaustive build the project has chosen instead of the demand-driven one.
"""

from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
df = pd.read_parquet(REPO / "registry" / "out" / "citations.parquet")

TOTAL = len(df)
print("citation rows: %d over %d documents\n" % (TOTAL, df.doc_id.nunique()))

print("=== where every row ended up ===")
for k, v in Counter(df.attribution).most_common():
    print("  %-20s %7d  %5.1f%%" % (k, v, 100 * v / TOTAL))

resolved = df[df.section_id.notna()]
strict = df[(df.attribution == "statute_span")
            | ((df.attribution == "nearest_preceding") & (df.distance <= 600))]

print("\n=== coverage curve (all attributed section citations) ===")
sec = Counter(resolved.section_id)
tot = sum(sec.values())
run, ranked = 0, sec.most_common()
marks = [0.50, 0.80, 0.90, 0.95, 0.99, 1.00]
mi = 0
for i, (_, c) in enumerate(ranked, 1):
    run += c
    while mi < len(marks) and run / tot >= marks[mi]:
        print("  %3d%% of citation mass <- top %5d sections" % (marks[mi] * 100, i))
        mi += 1

print("\n=== distinct sections cited, per act ===")
print("  (this is the floor for an exhaustive per-act build)")
per_act = resolved.groupby("act_id").agg(
    citations=("section_id", "size"), distinct_sections=("section_id", "nunique")
).sort_values("citations", ascending=False)
for act, row in per_act.iterrows():
    print("  %-22s %6d citations  %4d distinct sections"
          % (act, row.citations, row.distinct_sections))

crim = ["ipc-1860", "crpc-1973", "iea-1872"]
crim_cites = int(per_act.reindex(crim).citations.fillna(0).sum())
print("\ncriminal three (IPC/CrPC/IEA): %d of %d attributed citations (%.1f%%)"
      % (crim_cites, tot, 100 * crim_cites / tot))

print("\n=== the unresolved tail ===")
orph = df[df.attribution == "orphan"]
print("  orphan rows: %d (%.1f%% of all rows)" % (len(orph), 100 * len(orph) / TOTAL))
print("  these cite acts outside the 10-statute seed set; they are the")
print("  'unresolved, flagged, never silently dropped' bucket (PLAN.md M4)")
print("\n  most-cited orphan provisions:")
for t, c in Counter(orph.raw.str.lower().str.strip()).most_common(12):
    print("    %-24s %d" % (t[:24], c))
