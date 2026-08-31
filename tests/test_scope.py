"""Scope (I1) tests. test_cross_document_isolation must never be deleted or
skipped (CLAUDE.md §4)."""
from __future__ import annotations

import os
import tempfile

import pytest


def test_scope_rejects_empty_doc_ids():
    from smartlawai.scope import Scope
    with pytest.raises(ValueError):
        Scope(doc_ids=(), owner_id="alice")


def test_scope_rejects_empty_owner_id():
    from smartlawai.scope import Scope
    with pytest.raises(ValueError):
        Scope(doc_ids=("doc-1",), owner_id="")


def test_scope_is_frozen():
    from dataclasses import FrozenInstanceError

    from smartlawai.scope import Scope
    s = Scope(doc_ids=("doc-1",), owner_id="alice")
    with pytest.raises(FrozenInstanceError):
        s.owner_id = "bob"  # type: ignore[misc]


def test_scope_normalises_list_input_to_tuple():
    from smartlawai.scope import Scope
    s = Scope(doc_ids=["doc-1", "doc-2"], owner_id="alice")  # type: ignore[arg-type]
    assert s.doc_ids == ("doc-1", "doc-2")
    assert s.contains("doc-1")
    assert not s.contains("doc-3")


def _backend(tmp):
    os.environ["SMARTLAW_LOCAL_DIR"] = tmp
    from smartlawai.adapters.factory import get_backend
    return get_backend("local")


def test_cross_document_isolation():
    """A scoped fetch for doc A must never return doc B's chunks, and a scoped
    fetch for the right doc_id but wrong owner_id must also return empty --
    both dimensions of I1 enforced independently, not by construction alone."""
    from smartlawai.adapters.base import Chunk
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        be.store_chunks([
            Chunk(chunk_id="c-a1", doc_id="doc-a", chunk_index=0,
                 chunk_text="alice's contract clause", char_start=0, char_end=10,
                 owner_id="alice"),
            Chunk(chunk_id="c-b1", doc_id="doc-b", chunk_index=0,
                 chunk_text="bob's contract clause", char_start=0, char_end=10,
                 owner_id="bob"),
        ])

        # Right doc, right owner -> gets exactly doc A's chunk.
        result = be.fetch_chunks_scoped(Scope(doc_ids=("doc-a",), owner_id="alice"))
        assert [c.chunk_id for c in result] == ["c-a1"]

        # Right doc_id, wrong owner_id -> must return nothing, not doc A's chunk.
        result = be.fetch_chunks_scoped(Scope(doc_ids=("doc-a",), owner_id="bob"))
        assert result == []

        # Scoping to doc A must never leak doc B's chunk even under the correct
        # owner for doc A.
        result = be.fetch_chunks_scoped(Scope(doc_ids=("doc-a",), owner_id="alice"))
        assert all(c.doc_id == "doc-a" for c in result)
        assert "c-b1" not in [c.chunk_id for c in result]
