"""Confirm the NCRB correspondence tables parse cleanly as static HTML tables."""

import re
from html.parser import HTMLParser
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "registry" / "data" / "mha"


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.cur, self.cell, self.in_cell = [], [], [], False

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self.in_cell, self.cell = True, []
        elif tag == "tr":
            self.cur = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.in_cell = False
            self.cur.append(re.sub(r"\s+", " ", "".join(self.cell)).strip())
        elif tag == "tr":
            if any(c for c in self.cur):
                self.rows.append(self.cur)

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)


for name in ("BNS", "BNSS", "BSA"):
    p = TableParser()
    p.feed((DATA / (f"ncrb_{name}.html")).read_text(encoding="utf-8", errors="replace"))
    rows = p.rows
    widths = {}
    for r in rows:
        widths[len(r)] = widths.get(len(r), 0) + 1
    print("=== %s : %d rows, column-count histogram %s ===" % (name, len(rows), widths))
    for r in rows[:6]:
        print("   ", [c[:60] for c in r])
    print()
