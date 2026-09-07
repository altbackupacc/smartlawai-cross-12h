"""What granularity are SUMM's `document` list elements: sentences or paragraphs?"""

from pathlib import Path

from datasets import load_from_disk

REPO = Path(__file__).resolve().parents[2]
ds = load_from_disk(str(REPO / "data" / "raw" / "summ"))
rec = ds["test"][3]

doc = rec["document"]
print("id            :", rec.get("id"))
print("num_doc_tokens:", rec.get("num_doc_tokens"))
print("n elements    :", len(doc))
lens = [len(x) for x in doc]
print("elem chars    : min=%d median=%d max=%d" % (
    min(lens), sorted(lens)[len(lens) // 2], max(lens)))
print("\nfirst 6 elements:")
for i, x in enumerate(doc[:6]):
    print("  [%d] (%d chars) %s" % (i, len(x), x[:220].replace("\n", " ")))

# Where do act mentions sit relative to bare section refs, document-wide?
import sys

sys.path.insert(0, str(REPO))
from smartlawai.core.ner import extract_entities

text = "\n\n".join(doc)
res = extract_entities("probe", text)
st = sorted([e for e in res.entities if e.label == "STATUTE"], key=lambda e: e.start)
pr = sorted([e for e in res.entities if e.label == "PROVISION"], key=lambda e: e.start)
print("\nSTATUTE mentions: %d   PROVISION mentions: %d" % (len(st), len(pr)))
gaps = []
for p in pr:
    prev = [s for s in st if s.start < p.start]
    if prev:
        gaps.append(p.start - prev[-1].end)
if gaps:
    gaps.sort()
    print("char gap from provision back to nearest preceding STATUTE:")
    print("  min=%d  p25=%d  median=%d  p75=%d  max=%d" % (
        gaps[0], gaps[len(gaps)//4], gaps[len(gaps)//2], gaps[3*len(gaps)//4], gaps[-1]))
    for w in (200, 500, 1000, 2000, 5000):
        print("  within %5d chars: %d/%d (%.0f%%)" % (
            w, sum(1 for g in gaps if g <= w), len(pr), 100*sum(1 for g in gaps if g <= w)/len(pr)))
