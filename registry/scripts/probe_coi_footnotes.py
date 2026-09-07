"""Find how omission/repeal dates are written in the Constitution's footnotes."""

import re
from pathlib import Path

from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[2]
PDF = REPO / "registry" / "data" / "constitution" / "COI_english_2025-11.pdf"

r = PdfReader(str(PDF))


def is_english(t):
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 40:
        return False
    return sum(1 for c in letters if c.isascii()) / len(letters) >= 0.92


hits = 0
for i, page in enumerate(r.pages):
    t = page.extract_text() or ""
    if "Omitted by" not in t or not is_english(t):
        continue
    flat = re.sub(r"\s+", " ", t)
    for m in re.finditer(r"Omitted by[^.]{0,180}", flat):
        s = m.group(0)
        print("p%-4d %s" % (i + 1, s[:172]))
        hits += 1
        if hits > 26:
            raise SystemExit(0)
