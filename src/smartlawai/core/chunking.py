"""Clause-boundary chunker with scanning cursor and parent-child splitting
for context-preserving small-to-big retrieval (PLAN.md M2). Pure-Python, no heavy deps."""
from __future__ import annotations

import re
import uuid

from smartlawai import config
from smartlawai.adapters.base import Chunk

_BOUNDARY_MATCH = re.compile(
    r"^\s*(?P<label>Section\s+\d+[A-Za-z]?|Clause\s+\d+[A-Za-z]?|\d+\.|\(\d+\)|WHEREAS|NOW THEREFORE|IN WITNESS)",
    re.MULTILINE | re.IGNORECASE,
)


def _split_long(text: str, start: int, size: int) -> list[tuple[str, int, int]]:
    """Split long text into contiguous slices with exact character offsets."""
    out = []
    for i in range(0, len(text), size):
        seg = text[i:i + size]
        out.append((seg, start + i, start + i + len(seg)))
    return out


def chunk_document(doc_id: str, text: str, page_no: int | None = None) -> list[Chunk]:
    """Chunks legal text using a regex finditer scanning cursor.

    Key guarantees (M2):
    1. Exact character offsets even when clause text repeats (no re.split or str.find).
    2. Parent = full clause text without arbitrary truncation caps.
    3. Children = fine-grained slices (config.CHUNK_CHILD_CHARS) linked to parent_chunk_id.
    4. Small-to-big indexing: children are marked bm25_indexed=True; parents with children
       are marked bm25_indexed=False to prevent top-k duplicate crowding.
    5. Provenance metadata (page_no, para_no, section_label) populated on all chunks.
    """
    if not text:
        return []

    matches = list(_BOUNDARY_MATCH.finditer(text))
    spans: list[tuple[int, int, str | None]] = []

    if not matches:
        if text.strip():
            spans.append((0, len(text), None))
    else:
        # Leading preamble before first clause boundary
        if matches[0].start() > 0:
            preamble = text[0:matches[0].start()]
            if preamble.strip():
                spans.append((0, matches[0].start(), None))

        for i, m in enumerate(matches):
            span_start = m.start()
            span_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            label = m.group("label").strip() if m.group("label") else None
            spans.append((span_start, span_end, label))

    chunks: list[Chunk] = []
    idx = 0
    para_no = 1
    current_page = page_no if page_no is not None else 1

    for span_start, span_end, label in spans:
        piece = text[span_start:span_end]
        if not piece.strip():
            continue

        # Check for page feed characters within this span
        page_breaks = piece.count("\x0c") + piece.count("\f")
        if page_breaks > 0 and page_no is None:
            current_page += page_breaks

        # Exact trimmed offsets into original document text
        lead = len(piece) - len(piece.lstrip())
        trail = len(piece) - len(piece.rstrip())
        trimmed_start = span_start + lead
        trimmed_end = span_end - trail
        trimmed_text = text[trimmed_start:trimmed_end]

        parent_id = f"ch-{uuid.uuid4().hex[:10]}"
        has_children = len(trimmed_text) > config.CHUNK_CHILD_CHARS

        # Parent chunk: full clause text without truncation
        chunks.append(Chunk(
            chunk_id=parent_id,
            doc_id=doc_id,
            chunk_index=idx,
            chunk_text=trimmed_text,
            char_start=trimmed_start,
            char_end=trimmed_end,
            parent_chunk_id=None,
            bm25_indexed=not has_children,
            owner_id="",
            page_no=current_page,
            para_no=para_no,
            section_label=label,
        ))
        idx += 1

        # Child chunks for fine-grained retrieval
        if has_children:
            for seg, s, e in _split_long(trimmed_text, trimmed_start, config.CHUNK_CHILD_CHARS):
                chunks.append(Chunk(
                    chunk_id=f"ch-{uuid.uuid4().hex[:10]}",
                    doc_id=doc_id,
                    chunk_index=idx,
                    chunk_text=seg,
                    char_start=s,
                    char_end=e,
                    parent_chunk_id=parent_id,
                    bm25_indexed=True,
                    owner_id="",
                    page_no=current_page,
                    para_no=para_no,
                    section_label=label,
                ))
                idx += 1

        para_no += 1

    return chunks
