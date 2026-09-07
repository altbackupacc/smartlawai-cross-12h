"""Diagnose why PROVISION entities fail to attach to an act."""

from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
df = pd.read_parquet(REPO / "registry" / "out" / "citations.parquet")

print("rows by attribution:")
for k, v in Counter(df.attribution).most_common():
    print("  %-24s %d" % (k, v))

orph = df[df.attribution == "orphan"]
print("\norphan 'kind' distribution:")
for k, v in Counter(orph.kind).most_common():
    print("  %-10s %d" % (str(k), v))

print("\nmost common orphan raw strings:")
for t, c in Counter(orph.raw).most_common(30):
    print("  %-32s %d" % (t[:32], c))

# How far back is the nearest statute mention, really?
print("\nsample docs: statute mentions vs orphan provisions")
for doc_id, g in list(orph.groupby("doc_id"))[:5]:
    acts_in_doc = df[(df.doc_id == doc_id) & df.act_id.notna()]
    print("  %s: %d orphans, %d act mentions in doc (%s)"
          % (doc_id[:40], len(g), len(acts_in_doc),
             ",".join(sorted(set(acts_in_doc.act_id))[:4])))
