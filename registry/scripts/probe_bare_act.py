"""Probe a bare-act PDF's layout so the section parser can be written to fit."""

import re
import sys

from pypdf import PdfReader

path = sys.argv[1]
r = PdfReader(path)
print("pages:", len(r.pages))

# Dump a few pages from the middle, where the operative sections live.
for i in (3, 4, 12):
    if i >= len(r.pages):
        continue
    t = r.pages[i].extract_text() or ""
    print("\n================ page %d (%d chars) ================" % (i + 1, len(t)))
    print("\n".join(t.splitlines()[:38]))

# How many lines look like a section heading?
full = "\n".join((p.extract_text() or "") for p in r.pages)
pat = re.compile(r"^\s*(\d+[A-Z]{0,2})\.\s+(.{4,90})$", re.MULTILINE)
hits = pat.findall(full)
print("\n\n==== heading-shaped lines: %d ====" % len(hits))
for num, head in hits[:25]:
    print("  %-6s %s" % (num, head[:74]))
