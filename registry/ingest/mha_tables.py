"""Parse the official correspondence tables into SUPERSESSION rows.

Source: NCRB CyTrain HTML tables (see `registry/data/mha/PROVENANCE.md`).
These are the machine-readable form of the MHA correspondence tables published
when the three new criminal codes commenced on 2024-07-01.

`PLAN.md` M4: *"You are parsing and verifying, not deriving."* Nothing here
invents a mapping. Any row the parser cannot confidently interpret is recorded
as `unparsed` and reported, never dropped and never guessed (CLAUDE.md I2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "mha"

# The three codes commenced on this date.
COMMENCEMENT = date(2024, 7, 1)

# NCRB file -> (new act id, old act id)
TABLES = {
    "ncrb_BNS.html": ("bns-2023", "ipc-1860"),
    "ncrb_BNSS.html": ("bnss-2023", "crpc-1973"),
    "ncrb_BSA.html": ("bsa-2023", "iea-1872"),
}

# Cells that mean "there is no counterpart", not "here is a section number".
_NEW_MARKERS = re.compile(r"\bnew\s+(section|sub-?section|provision|clause)\b", re.IGNORECASE)
_REPEALED_MARKERS = re.compile(r"\brepealed\b|\bomitted\b|\bdeleted\b", re.IGNORECASE)
_CHAPTER_ROW = re.compile(r"^\s*(part|chapter|schedule)\b", re.IGNORECASE)

# A section number introducing a heading: "302.", "304A.", "14- ", "41A Notice".
# The trailing separator (period, dash, or space-then-capital) is required so
# that stray numbers inside prose ("within 30 days") are not read as citations.
# A bare trailing period alone is too strict: the tables also use "14-" and
# "41A Notice of appearance" with no punctuation at all.
_SEC_NUM = re.compile(
    r"(?<![\w(])(\d+[A-Z]{0,2})\s*(?:\((\d+[A-Za-z]?)\))?"
    r"(?=\s*[.\-–—]|\s+[A-Z“‘\"'])"
)
# Sub-section form, possibly followed by the defined term: "2(3)", "2(3) 'child'".
_SUBSEC_ONLY = re.compile(r"^\s*(\d+[A-Z]{0,2})\s*\((\d+[A-Za-z]?)\)")
# Words that mark the number after them as a cross-reference to another
# provision rather than this row's own heading number.
_XREF_PREFIX = re.compile(r"(?:sections?|act|schedule|chapter|clause)\s+(?:no\.?\s*)?$", re.IGNORECASE)


@dataclass
class MappingRow:
    """One parsed correspondence row."""

    new_act: str
    old_act: str
    new_sections: list[str] = field(default_factory=list)
    old_sections: list[str] = field(default_factory=list)
    relation: str = "maps_to"     # maps_to | new_provision | repealed_no_successor
    raw_new: str = ""
    raw_old: str = ""


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._cur: list[str] = []
        self._cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag in ("td", "th"):
            self._in_cell, self._cell = True, []
        elif tag == "tr":
            self._cur = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._in_cell = False
            self._cur.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
        elif tag == "tr":
            if any(c for c in self._cur):
                self.rows.append(self._cur)

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)


def _is_year(num: str) -> bool:
    """1860, 1974, 2023 ... no statute in scope has a section numbered like a year.

    Cells routinely name other statutes inline -- "under section 84 of
    Bharatiya Nagarik Suraksha Sanhita, 2023" and "of Act 2 of 1974" -- and the
    trailing year otherwise parses as a section, then forms a cross-product of
    fabricated mappings with everything else in the row.
    """
    return num.isdigit() and 1800 <= int(num) <= 2100


def _section_spans(cell: str) -> list[tuple[str, int, int]]:
    """Operative section numbers in a cell as (number, match_start, match_end).

    Only *operative* numbers count -- a number introducing its own heading.
    Numbers appearing as cross-references inside prose ("under section 84 of
    ...", "Act 2 of 1974") are references to other provisions, not a statement
    that this row maps them.
    """
    if not cell:
        return []
    m = _SUBSEC_ONLY.match(cell)
    if m:
        return [(m.group(1), m.start(), m.end())]
    out, seen = [], set()
    for mm in _SEC_NUM.finditer(cell):
        num = mm.group(1).upper()
        if _is_year(num):
            continue
        prefix = cell[max(0, mm.start() - 14):mm.start()].lower()
        if _XREF_PREFIX.search(prefix):
            continue
        if num not in seen:
            seen.add(num)
            out.append((num, mm.start(), mm.end()))
    return out


def _section_numbers(cell: str) -> list[str]:
    """Every operative section number in a cell, in order, deduplicated.

    Sub-sections are folded to their parent section: the registry tracks
    in-force status at section granularity (see `normalise.section_id`), so
    BNS 1(3) is recorded against BNS section 1.
    """
    return [n for n, _, _ in _section_spans(cell)]


# Leading punctuation/dash noise the tables use between number and heading.
_HEAD_LEAD = re.compile(r"^[\s.\-–—:;,―—–]+")
_HEAD_TRAIL = re.compile(r"[\s.\-–—:;,]+$")


def section_headings(cell: str) -> dict[str, str]:
    """Map each operative section number in a cell to its heading text.

    The old-code column of the correspondence tables carries the repealed
    provisions' headings -- "174A. Non-appearance in response to a
    proclamation..." -- which is the only machine-readable source we have for
    IPC/CrPC/IEA headings, since India Code does not index the repealed codes.

    A heading runs from the end of its number to the start of the next one.
    """
    spans = _section_spans(cell)
    out: dict[str, str] = {}
    for idx, (num, _start, end) in enumerate(spans):
        stop = spans[idx + 1][1] if idx + 1 < len(spans) else len(cell)
        text = cell[end:stop]
        text = _HEAD_LEAD.sub("", text)
        text = _HEAD_TRAIL.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Drop the markers the tables use in place of a heading.
        if not text or _NEW_MARKERS.fullmatch(text) or text.lower() in {
            "repealed", "deleted", "omitted", "new section", "new sub-section",
        }:
            continue
        if len(text) >= 3:
            out[num] = text
    return out


def parse_table(path: Path, new_act: str, old_act: str) -> tuple[list[MappingRow], dict]:
    """Parse one NCRB correspondence table."""
    p = _TableParser()
    p.feed(path.read_text(encoding="utf-8", errors="replace"))

    rows: list[MappingRow] = []
    stats = {
        "total_rows": 0, "header": 0, "chapter": 0, "mapped": 0,
        "new_provision": 0, "repealed_no_successor": 0, "unparsed": 0,
    }

    for r in p.rows:
        stats["total_rows"] += 1
        if len(r) != 2:
            stats["unparsed"] += 1
            continue
        left, right = r[0].strip(), r[1].strip()

        # Header row names the two acts.
        if ("sanhita" in left.lower() or "adhiniyam" in left.lower()) and (
            "penal code" in right.lower() or "criminal procedure" in right.lower()
            or "evidence act" in right.lower()
        ):
            stats["header"] += 1
            continue

        if _CHAPTER_ROW.match(left) or _CHAPTER_ROW.match(right):
            stats["chapter"] += 1
            continue

        new_secs = _section_numbers(left)
        old_secs = _section_numbers(right)

        # "New Section" on the right: the new code added it, nothing precedes it.
        if _NEW_MARKERS.search(right) and not old_secs:
            if not new_secs:
                stats["unparsed"] += 1
                continue
            rows.append(MappingRow(new_act, old_act, new_secs, [],
                                   "new_provision", left, right))
            stats["new_provision"] += 1
            continue

        # Empty left cell, or an explicit "Deleted"/"Repealed" marker on EITHER
        # side: the old provision has no counterpart in the new code. The marker
        # is usually in the left (new-code) cell -- "Deleted" opposite
        # "124A. Sedition" is how the tables record sedition's repeal, and
        # checking only the right cell misses every such row.
        if old_secs and (not new_secs) and (
            not left or _REPEALED_MARKERS.search(left) or _REPEALED_MARKERS.search(right)
        ):
            rows.append(MappingRow(new_act, old_act, [], old_secs,
                                   "repealed_no_successor", left, right))
            stats["repealed_no_successor"] += 1
            continue

        if new_secs and old_secs:
            rows.append(MappingRow(new_act, old_act, new_secs, old_secs,
                                   "maps_to", left, right))
            stats["mapped"] += 1
            continue

        stats["unparsed"] += 1

    return rows, stats


def parse_all() -> tuple[list[MappingRow], dict[str, dict]]:
    all_rows, all_stats = [], {}
    for fname, (new_act, old_act) in TABLES.items():
        path = DATA / fname
        if not path.exists():
            all_stats[fname] = {"error": "missing"}
            continue
        rows, stats = parse_table(path, new_act, old_act)
        all_rows.extend(rows)
        all_stats[fname] = stats
    return all_rows, all_stats
