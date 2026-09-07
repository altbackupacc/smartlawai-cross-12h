"""End-to-end demo: raw citation -> registry target -> authority verdict."""

import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.resolver import resolve_citation
from verify.registry_check import check_citation

TODAY = date(2026, 9, 5)
CASES = [
    "Section 420 IPC",
    "Section 302 of the Indian Penal Code",
    "Section 124A IPC",
    "Section 154 CrPC",
    "Section 65B of the Indian Evidence Act",
    "Section 318 BNS",
    "Section 138 of the Negotiable Instruments Act",
    "Indian Penal Code",
    "Section 9999 IPC",
    "Article 21 of the Constitution of India",
    "Section 5 of the Made Up Act 2019",
]

print(f"as-of date: {TODAY}\n")
print("%-46s %-12s %s" % ("CITATION", "STATUS", "NOTE"))
print("-" * 110)
for raw in CASES:
    cit = resolve_citation(raw)
    if cit is None:
        print("%-46s %-12s %s" % (raw[:46], "UNRESOLVED", "not in registry - flagged, gate strikes the claim"))
        continue
    res = check_citation(cit, as_of=TODAY)
    print("%-46s %-12s %s" % (raw[:46], res.status, res.note or ""))

print("\n--- the same citation, judged as of 2020-01-01 ---")
for raw in ("Section 420 IPC", "Section 318 BNS"):
    cit = resolve_citation(raw)
    res = check_citation(cit, as_of=date(2020, 1, 1))
    print("%-46s %-12s %s" % (raw, res.status, res.note or ""))
