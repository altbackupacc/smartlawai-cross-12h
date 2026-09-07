"""Probe the official Constitution PDF to design the article parser."""

import re
from pathlib import Path

from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[2]
PDF = REPO / "registry" / "data" / "constitution" / "COI_legislative_gov_in_2026-02.pdf"

r = PdfReader(str(PDF))
print("pages:", len(r.pages))

# Where does ARRANGEMENT OF ARTICLES start, and where does the body start?
for i in range(min(40, len(r.pages))):
    t = (r.pages[i].extract_text() or "")
    up = t.upper()
    if "ARRANGEMENT OF ARTICLES" in up:
        print("ARRANGEMENT OF ARTICLES on page", i + 1)
    if re.search(r"\bPART\s+I\b", up) and "UNION AND ITS TERRITORY" in up:
        print("PART I (body/toc) marker on page", i + 1)

for pg in (5, 6, 7, 12):
    if pg >= len(r.pages):
        continue
    t = r.pages[pg].extract_text() or ""
    print("\n=========== page %d (%d chars) ===========" % (pg + 1, len(t)))
    for ln in t.splitlines()[:32]:
        print("   ", repr(ln[:110]))

# Is there Devanagari (bilingual edition)?
full_sample = "".join((r.pages[i].extract_text() or "") for i in range(30))
dev = sum(1 for ch in full_sample if "ऀ" <= ch <= "ॿ")
print("\nDevanagari chars in first 30 pages:", dev)
