"""Week-1 triage: is a comparison-table PDF machine-readable or a scan?

PLAN.md calls this "the largest single unknown in the whole estimate":
parseable -> ~2h, scanned -> ~20h.
"""
import sys

from pypdf import PdfReader

path = sys.argv[1]
r = PdfReader(path)
print("pages:", len(r.pages))

total_chars = 0
total_images = 0
for i, pg in enumerate(r.pages):
    t = pg.extract_text() or ""
    total_chars += len(t)
    try:
        total_images += len(pg.images)
    except Exception:
        pass
    if i < 2:
        print(f"\n--- page {i+1}: {len(t)} chars ---")
        print("\n".join(t.splitlines()[:25]))

print("\n==== TOTALS ====")
print("chars extracted :", total_chars)
print("images embedded :", total_images)
print("chars/page      :", round(total_chars / max(len(r.pages), 1)))
verdict = "MACHINE-READABLE" if total_chars / max(len(r.pages), 1) > 200 else "LIKELY SCANNED"
print("VERDICT         :", verdict)
