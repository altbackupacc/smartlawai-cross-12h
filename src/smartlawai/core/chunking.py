"""Clause-boundary chunker with parent-child splitting for context-preserving
fine-grained retrieval. Pure-Python, no heavy deps."""
from __future__ import annotations

import re
import uuid

from smartlawai.adapters.base import Chunk

_BOUNDARY = re.compile(
    r"(?=^\s*(?:\d+\.|\(\d+\)|Section\s+\d+|Clause\s+\d+|WHEREAS|NOW THEREFORE|"
    r"IN WITNESS))",
    re.MULTILINE | re.IGNORECASE,
)
_MAX_CHARS = 1200   # parent chunk cap
_CHILD_CHARS = 400  # fine-grained child size


def _split_long(text: str, start: int, size: int) -> list[tuple[str, int, int]]:
    out = []
    for i in range(0, len(text), size):
        seg = text[i:i + size]
        out.append((seg, start + i, start + i + len(seg)))
    return out


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    pieces = _BOUNDARY.split(text)
    chunks: list[Chunk] = []
    idx = 0
    cursor = 0
    for piece in pieces:
        piece = piece.strip("\n")
        if not piece.strip():
            cursor += len(piece)
            continue
        p_start = text.find(piece, cursor)
        if p_start < 0:
            p_start = cursor
        cursor = p_start + len(piece)

        parent_id = f"ch-{uuid.uuid4().hex[:10]}"
        chunks.append(Chunk(
            chunk_id=parent_id, doc_id=doc_id, chunk_index=idx,
            chunk_text=piece[:_MAX_CHARS], char_start=p_start,
            char_end=p_start + min(len(piece), _MAX_CHARS), parent_chunk_id=None))
        idx += 1

        if len(piece) > _CHILD_CHARS:
            for seg, s, e in _split_long(piece, p_start, _CHILD_CHARS):
                chunks.append(Chunk(
                    chunk_id=f"ch-{uuid.uuid4().hex[:10]}", doc_id=doc_id,
                    chunk_index=idx, chunk_text=seg, char_start=s, char_end=e,
                    parent_chunk_id=parent_id))
                idx += 1
    return chunks
