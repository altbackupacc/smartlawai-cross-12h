"""Ingest the Constitution's articles from the official Legislative Department text.

Source: `THE CONSTITUTION OF INDIA [As on 11th November, 2025]`, Ministry of Law
and Justice, Legislative Department, via legislative.gov.in. The Constitution is
**not** in India Code -- it is not an "Act" in that database's model -- so the
official PDF is the primary source available.

The edition retrieved is an English-Kannada *diglot*: English and Kannada
alternate page by page, with the Kannada in a legacy encoding that extracts as
Latin-1 mojibake. Parsing therefore filters to English pages by ASCII ratio
before matching; without that filter the Kannada column silently supplies
headings for articles whose English entry did not match, which is how Article
31's heading first came back as Kannada text.

Two entry shapes appear in the arrangement of articles:

    32. Remedies for enforcement of rights conferred by this Part ... 19
    [31. Omitted.] ................................................. 16

The bracketed form is the official marker for a repealed/omitted article. It
matters here more than anywhere else in the registry: Article 31 (right to
property) was repealed by the 44th Amendment in 1978 and is cited 443 times in
the corpus, making it a second instance of exactly the staleness the project
measures -- see `PLAN.md`'s V2 note that a further supersession wave needs no
schema change.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "constitution"
DEFAULT_PDF = DATA / "COI_english_2025-11.pdf"

# "[31. Omitted.]" / "[32A. Omitted]" -- official repeal marker.
OMITTED = re.compile(r"\[\s*(\d+[A-Z]{0,3})\.\s*Omitted\.?\s*\]", re.IGNORECASE)

# "32. Remedies for ... this Part ... 19" -- heading then dot leaders then page.
ENTRY = re.compile(
    r"(?<![\d.\[])(\d+[A-Z]{0,3})\.\s+"      # article number
    r"([A-Z][^\[\]]{3,220}?)\s*"             # heading, starts with a capital
    r"[.…]{2,}\s*\d"                    # dot leaders (or ellipsis) then a page no.
)

ASCII_MIN = 0.92  # English pages; Kannada pages fall far below this


@dataclass
class Article:
    number: str
    heading: str | None
    omitted: bool = False
    omitted_on: dt.date | None = None
    omitted_by: str | None = None


# Body form of an omitted article. It carries BOTH the original heading (in
# square brackets) and the effective date, e.g.
#   31. [Compulsory acquisition of property.].— Omitted by the Constitution
#       (Forty-fourth Amendment) Act, 1978, s. 6 (w.e.f. 20-6-1979).
OMIT_BODY = re.compile(
    r"(?<![\d.])(\d+[A-Z]{0,3})\.\s*\[\s*([^\]]{3,180}?)\s*\.?\s*\]"   # num + [heading]
    r"[\s.—–-]*"                                             # dashes/dots
    r"(?:Omitted|Rep\.)\s+by\s+"                                       # the marker
    r"(.{0,150}?)"                                                     # amending Act
    r"\(\s*w\.e\.f\.\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{4})\s*\)",  # effective date
    re.IGNORECASE | re.DOTALL,
)


# The Constitution's highest article number is 395. Anything above that in the
# body text is a footnote superscript that PDF extraction has glued onto the
# article number -- "3329A." is footnote 3 followed by article 329A. Left
# uncorrected these become phantom articles and the real ones keep no date.
MAX_ARTICLE = 395


def _strip_footnote_prefix(num: str, known: set[str] | None = None) -> str | None:
    """'3329A' -> '329A'. Returns None if no plausible article number remains.

    `known` is the article set from the arrangement of articles, and it is what
    resolves the genuinely ambiguous cases. "132A" is a valid-looking number, so
    a magnitude check alone accepts it -- but in the body it is footnote 1
    followed by article 32A, and article 132A does not exist. Preferring a
    candidate that the official article list actually contains settles it.
    """
    candidates = []
    cur = num
    for _ in range(3):
        m = re.match(r"(\d+)([A-Z]{0,3})$", cur)
        if not m:
            break
        candidates.append(cur)
        stripped = m.group(1)[1:] + m.group(2)
        if not stripped or not stripped[0].isdigit():
            break
        cur = stripped

    if known:
        for c in candidates:
            if c in known:
                return c
    for c in candidates:
        m = re.match(r"(\d+)", c)
        if m and int(m.group(1)) <= MAX_ARTICLE:
            return c
    return None


def _is_english_page(text: str) -> bool:
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 40:
        return False
    ascii_letters = sum(1 for c in letters if c.isascii())
    return ascii_letters / len(letters) >= ASCII_MIN


def parse_articles(pdf_path: Path | None = None) -> tuple[dict[str, Article], dict]:
    from pypdf import PdfReader

    path = Path(pdf_path or DEFAULT_PDF)
    reader = PdfReader(str(path))

    arts: dict[str, Article] = {}
    stats = {"pages": len(reader.pages), "english_pages": 0, "skipped_pages": 0,
             "omitted": 0, "with_heading": 0}

    for page in reader.pages:
        raw = page.extract_text() or ""
        if "." * 3 not in raw and "…" not in raw:
            continue
        if not _is_english_page(raw):
            stats["skipped_pages"] += 1
            continue
        stats["english_pages"] += 1
        flat = re.sub(r"\s+", " ", raw)

        for num in OMITTED.findall(flat):
            num = num.upper()
            prev = arts.get(num)
            if prev is None or not prev.omitted:
                arts[num] = Article(number=num, heading=None, omitted=True)

        for num, head in ENTRY.findall(flat):
            num = num.upper()
            head = re.sub(r"\s+", " ", head).strip(" .…")
            if len(head) < 4:
                continue
            if num in arts and arts[num].omitted:
                continue          # an explicit Omitted marker wins
            if num not in arts or not arts[num].heading:
                arts[num] = Article(number=num, heading=head, omitted=False)

    # --- second pass: recover omission dates (and original headings) from the
    # body. The arrangement of articles states only *that* an article was
    # omitted; the amending Act and its effective date live in the body entry.
    # Without this pass the registry can strike a repealed article but cannot
    # say when it ceased to apply, which makes an as-of query before the repeal
    # unanswerable.
    dates = parse_omission_dates(reader, known=set(arts))
    for num, (when, by_whom, heading) in dates.items():
        a = arts.get(num)
        if a is None:
            a = Article(number=num, heading=heading, omitted=True)
            arts[num] = a
        a.omitted = True
        a.omitted_on = when
        a.omitted_by = by_whom
        if not a.heading and heading:
            a.heading = heading

    stats["omitted"] = sum(1 for a in arts.values() if a.omitted)
    stats["omitted_with_date"] = sum(1 for a in arts.values() if a.omitted_on)
    stats["with_heading"] = sum(1 for a in arts.values() if a.heading)
    return arts, stats


def parse_omission_dates(reader, known: set[str] | None = None) -> dict[str, tuple[dt.date, str, str]]:
    """number -> (effective date, amending Act, original heading)."""
    out: dict[str, tuple[dt.date, str, str]] = {}
    for page in reader.pages:
        raw = page.extract_text() or ""
        if "w.e.f." not in raw:
            continue
        if not _is_english_page(raw):
            continue
        flat = re.sub(r"\s+", " ", raw)
        for m in OMIT_BODY.finditer(flat):
            num = _strip_footnote_prefix(m.group(1).upper(), known)
            if num is None:
                continue
            heading = re.sub(r"\s+", " ", m.group(2)).strip(" .")
            by_whom = re.sub(r"\s+", " ", m.group(3)).strip(" ,.")
            d, mo, y = int(m.group(4)), int(m.group(5)), int(m.group(6))
            try:
                when = dt.date(y, mo, d)
            except ValueError:
                continue
            out.setdefault(num, (when, by_whom, heading))
    return out
