"""Unit tests for core/chunking.py (PLAN.md M2).
Validates scanning cursor offsets, repeated text handling, parent/child small-to-big splitting,
and provenance metadata."""
from __future__ import annotations

from smartlawai import config
from smartlawai.core.chunking import chunk_document


def test_chunking_exact_offsets_on_repeated_text():
    """Identical clauses must receive distinct, strictly increasing character offsets
    that exactly match slicing into the original text."""
    repeated_clause = "This Agreement shall be governed by the laws of India.\nNeither party may assign this agreement."
    doc_text = f"Section 1. Jurisdiction\n{repeated_clause}\n\nSection 2. Miscellaneous\n{repeated_clause}"

    chunks = chunk_document("doc-test", doc_text)
    assert len(chunks) >= 2

    parents = [c for c in chunks if c.parent_chunk_id is None]
    assert len(parents) == 2

    p1, p2 = parents[0], parents[1]
    assert p1.section_label == "Section 1"
    assert p2.section_label == "Section 2"

    # Offsets must be strictly monotonic
    assert p1.char_start < p1.char_end <= p2.char_start < p2.char_end

    # Slicing the raw text with offsets must yield the chunk text exactly
    assert doc_text[p1.char_start:p1.char_end] == p1.chunk_text
    assert doc_text[p2.char_start:p2.char_end] == p2.chunk_text


def test_parent_has_full_clause_no_truncation():
    """Parent chunk must not be arbitrarily truncated to 1200 characters."""
    long_clause_body = "The vendor shall indemnify and hold harmless the purchaser against all liabilities. " * 30
    doc_text = f"Clause 10. Indemnification\n{long_clause_body}"
    assert len(doc_text) > 1500

    chunks = chunk_document("doc-long", doc_text)
    parents = [c for c in chunks if c.parent_chunk_id is None]
    assert len(parents) == 1
    parent = parents[0]

    # Verify full text is retained
    assert len(parent.chunk_text) > 1500
    assert doc_text[parent.char_start:parent.char_end] == parent.chunk_text


def test_parent_child_linking_and_indexing():
    """Long clauses must produce child chunks with valid parent_chunk_id and correct indexing flags."""
    long_clause_body = "Important legal terms and conditions specified herein. " * 20
    doc_text = f"Section 5. Conditions\n{long_clause_body}"

    chunks = chunk_document("doc-split", doc_text)
    parents = [c for c in chunks if c.parent_chunk_id is None]
    children = [c for c in chunks if c.parent_chunk_id is not None]

    assert len(parents) == 1
    assert len(children) > 1

    parent = parents[0]
    # Small-to-big indexing rule: parent with children is not directly indexed
    assert parent.bm25_indexed is False

    for child in children:
        assert child.parent_chunk_id == parent.chunk_id
        assert child.bm25_indexed is True
        assert child.section_label == "Section 5"
        # Child slice matches original text
        assert doc_text[child.char_start:child.char_end] == child.chunk_text


def test_short_clause_indexed_directly():
    """Short clauses that do not need splitting must have bm25_indexed=True on parent."""
    doc_text = "Section 99. Short\nValid notice by registered post."
    assert len(doc_text) < config.CHUNK_CHILD_CHARS

    chunks = chunk_document("doc-short", doc_text)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.parent_chunk_id is None
    assert c.bm25_indexed is True
    assert c.section_label == "Section 99"


def test_provenance_metadata():
    """page_no and para_no must be populated."""
    doc_text = "Section 1. First\nText 1.\n\nSection 2. Second\nText 2."
    chunks = chunk_document("doc-prov", doc_text, page_no=5)

    parents = [c for c in chunks if c.parent_chunk_id is None]
    assert len(parents) == 2
    assert parents[0].page_no == 5
    assert parents[0].para_no == 1
    assert parents[1].page_no == 5
    assert parents[1].para_no == 2
