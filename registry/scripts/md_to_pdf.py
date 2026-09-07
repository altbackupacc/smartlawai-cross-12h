"""Render a Markdown work log to PDF via reportlab Platypus.

Handles the subset of Markdown used in M4_WORKLOG.md: ATX headings, paragraphs,
fenced code blocks, pipe tables, bullet lists, horizontal rules, and inline
bold / italic / code.

Fonts are registered from Windows TTFs rather than using reportlab's built-in
Type1 faces, because the built-ins lack glyphs for the arrows, section signs and
em-dashes in this document and would render them as solid black boxes.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "consola.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold")

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6470")
RULE = colors.HexColor("#d6dae0")
CODE_BG = colors.HexColor("#f5f6f8")
HEAD = colors.HexColor("#12263f")

ss = getSampleStyleSheet()
S = {
    "h1": ParagraphStyle("h1", parent=ss["Title"], fontName="Body-Bold", fontSize=21,
                         leading=26, textColor=HEAD, spaceAfter=4, alignment=TA_LEFT),
    "h2": ParagraphStyle("h2", fontName="Body-Bold", fontSize=14.5, leading=19,
                         textColor=HEAD, spaceBefore=16, spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="Body-Bold", fontSize=11.5, leading=15,
                         textColor=HEAD, spaceBefore=11, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Body", fontSize=9.4, leading=13.6,
                        textColor=INK, spaceAfter=6),
    "sub": ParagraphStyle("sub", fontName="Body", fontSize=9.4, leading=13.6,
                          textColor=MUTED, spaceAfter=2),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.0, leading=10.8,
                           textColor=INK, backColor=CODE_BG,
                           borderPadding=(6, 6, 6, 6), spaceBefore=4, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.1, leading=11,
                           textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="Body-Bold", fontSize=8.1, leading=11,
                            textColor=colors.white),
    "li": ParagraphStyle("li", fontName="Body", fontSize=9.4, leading=13.4,
                         textColor=INK, spaceAfter=2),
}


def inline(t: str) -> str:
    """Markdown inline -> reportlab mini-HTML."""
    t = html.escape(t, quote=False)
    t = re.sub(r"`([^`]+)`", r'<font face="Mono" size="8.4">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
    return t


def split_row(line: str) -> list[str]:
    line = line.strip()
    line = line.removeprefix("|")
    line = line.removesuffix("|")
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)]


def build_table(rows: list[list[str]], width: float) -> Table:
    header, body = rows[0], rows[1:]
    ncol = len(header)
    data = [[Paragraph(inline(c), S["cellh"]) for c in header]]
    for r in body:
        r = (r + [""] * ncol)[:ncol]
        data.append([Paragraph(inline(c), S["cell"]) for c in r])

    # First column usually holds identifiers; give it a little more room.
    if ncol == 2:
        cw = [width * 0.42, width * 0.58]
    elif ncol == 3:
        cw = [width * 0.30, width * 0.20, width * 0.50]
    else:
        first = width * 0.26
        cw = [first] + [(width - first) / (ncol - 1)] * (ncol - 1)

    t = Table(data, colWidths=cw, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def convert(md_path: Path, pdf_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=16 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title="M4 - Authority Registry Work Log", author="SmartLawAI",
    )
    avail = doc.width
    story = []
    i = 0
    bullets: list[str] = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            story.append(ListFlowable(
                [ListItem(Paragraph(inline(b), S["li"]), leftIndent=12) for b in bullets],
                bulletType="bullet", start="•", leftIndent=14,
            ))
            story.append(Spacer(1, 5))
            bullets = []

    while i < len(lines):
        ln = lines[i]

        # fenced code
        if ln.lstrip().startswith("```"):
            flush_bullets()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            body = html.escape("\n".join(buf)).replace(" ", "&nbsp;").replace("\n", "<br/>")
            story.append(Paragraph(body, S["code"]))
            continue

        # table
        if ln.strip().startswith("|") and i + 1 < len(lines) and re.match(
                r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
            flush_bullets()
            rows = [split_row(ln)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            story.append(build_table(rows, avail))
            story.append(Spacer(1, 8))
            continue

        s = ln.strip()

        if not s:
            flush_bullets()
            i += 1
            continue

        if s in ("---", "***", "___"):
            flush_bullets()
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE))
            story.append(Spacer(1, 5))
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush_bullets()
            level = len(m.group(1))
            text = inline(m.group(2))
            key = "h1" if level == 1 else ("h2" if level == 2 else "h3")
            para = Paragraph(text, S[key])
            if key == "h2":
                story.append(KeepTogether([para, Spacer(1, 1)]))
            else:
                story.append(para)
            i += 1
            continue

        m = re.match(r"^[-*]\s+(.*)$", s)
        if m:
            bullets.append(m.group(1))
            i += 1
            continue

        m = re.match(r"^\d+\.\s+(.*)$", s)
        if m:
            bullets.append(m.group(1))
            i += 1
            continue

        if s.startswith(">"):
            flush_bullets()
            story.append(Paragraph(inline(s.lstrip("> ").strip()), S["sub"]))
            i += 1
            continue

        # paragraph: join wrapped lines
        buf = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", "```", "- ", "* ", ">"))
                    or nxt in ("---", "***", "___") or re.match(r"^\d+\.\s", nxt)):
                break
            buf.append(nxt)
            i += 1
        flush_bullets()
        story.append(Paragraph(inline(" ".join(buf)), S["p"]))

    flush_bullets()

    def footer(canv, _doc):
        canv.saveState()
        canv.setFont("Body", 7.5)
        canv.setFillColor(MUTED)
        canv.drawString(18 * mm, 9 * mm, "SmartLawAI - M4 Authority Registry - work log")
        canv.drawRightString(A4[0] - 16 * mm, 9 * mm, "Page %d" % canv.getPageNumber())
        canv.setStrokeColor(RULE)
        canv.line(18 * mm, 12.5 * mm, A4[0] - 16 * mm, 12.5 * mm)
        canv.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    convert(src, dst)
    print(f"wrote {dst} ({dst.stat().st_size / 1024:.1f} KB)")
