"""Locate and test-parse the Constitution's ARRANGEMENT OF ARTICLES.

TOC entries wrap across lines -- the heading may break mid-phrase with the dot
leaders and page numbers on following lines -- so the page text is joined and
whitespace-collapsed before matching.
"""

import json
import re
from pathlib import Path

from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[2]
PDF = REPO / "registry" / "data" / "constitution" / "COI_english_2025-11.pdf"
OUT = REPO / "registry" / "out" / "coi_toc.json"

r = PdfReader(str(PDF))

# "128. Attendance of retired Judges at sittings of the Supreme Court ....... 60-61"
ENTRY = re.compile(r"(?<![\d.])(\d+[A-Z]{0,3})\.\s+([^.]{4,300}?)\s*\.{4,}")

found: dict[str, str] = {}
toc_pages = []

for i in range(len(r.pages)):
    raw = r.pages[i].extract_text() or ""
    if ".." not in raw:
        continue
    flat = re.sub(r"\s+", " ", raw)
    hits = ENTRY.findall(flat)
    if not hits:
        continue
    toc_pages.append((i + 1, len(hits)))
    for num, head in hits:
        head = re.sub(r"\s+", " ", head).strip(" .")
        if len(head) >= 3:
            found.setdefault(num.upper(), head)

print("TOC-ish pages: %d  (first few: %s)" % (len(toc_pages), toc_pages[:8]))
print("distinct article numbers: %d" % len(found))

ks = sorted(found, key=lambda s: (int(re.match(r"\d+", s).group()), s))
print("first 10:", [(k, found[k][:44]) for k in ks[:10]])
print("last  6 :", [(k, found[k][:44]) for k in ks[-6:]])
print()
for probe in ("14", "19", "21", "31", "32", "51A", "136", "226", "311", "370", "395"):
    print("  art %-4s -> %s" % (probe, found.get(probe, "<<MISSING>>")[:70]))

OUT.write_text(json.dumps(found, indent=2), encoding="utf-8")
print(f"\nwrote {OUT}")
