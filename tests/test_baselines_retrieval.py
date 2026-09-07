"""Tests for baselines/retrieval.py -- I1: every retrieving function must be
scoped and must never fall back to fetch_all_chunks(). Fake encoder/reranker
are used instead of real InLegalBERT/CrossEncoder so this test doesn't need
the heavy `ml` extras installed."""
from __future__ import annotations

import os
import tempfile
from unittest import mock

import numpy as np


def _backend(tmp):
    os.environ["SMARTLAW_LOCAL_DIR"] = tmp
    from smartlawai.adapters.factory import get_backend
    return get_backend("local")


def _ingest(be, text, owner_id="alice"):
    from smartlawai.pipeline import Pipeline
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        path = f.name
    doc_id, _ = Pipeline(be).ingest(path, doc_type="JUDGMENT", source="test",
                                    owner_id=owner_id)
    return doc_id


class _FakeEncoder:
    """Deterministic, dependency-free stand-in for InLegalBERTEncoder --
    scoped_hybrid only needs .embed(list[str]) -> L2-normalised vectors."""

    def embed(self, texts):
        vecs = []
        for t in texts:
            v = np.zeros(8, dtype="float32")
            for i, ch in enumerate(t.lower()):
                v[i % 8] += ord(ch)
            n = np.linalg.norm(v)
            vecs.append(v / n if n else v)
        return np.vstack(vecs) if vecs else np.zeros((0, 8), dtype="float32")


class _FakeReranker:
    def rerank(self, query, passages, top_k):
        return [(cid, 1.0 - i * 0.01) for i, (cid, _) in enumerate(passages[:top_k])]


def test_scoped_bm25_returns_only_in_scope_chunks():
    from baselines.retrieval import scoped_bm25
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        doc_a = _ingest(be, "Section 74 of the Indian Contract Act, 1872 governs damages.")
        _ingest(be, "The Arbitration and Conciliation Act, 1996 applies here.")

        result = scoped_bm25(be, Scope(doc_ids=(doc_a,), owner_id="alice"), "damages")
        assert result
        assert all(rc.chunk.doc_id == doc_a for rc in result)


def test_scoped_bm25_empty_for_nonmatching_scope():
    from baselines.retrieval import scoped_bm25
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        assert scoped_bm25(be, Scope(doc_ids=("doc-nonexistent",), owner_id="alice"),
                           "anything") == []


def test_scoped_bm25_never_calls_fetch_all_chunks():
    from baselines.retrieval import scoped_bm25
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        doc_a = _ingest(be, "Section 74 of the Indian Contract Act, 1872 governs damages.")
        with mock.patch.object(type(be), "fetch_all_chunks",
                               side_effect=AssertionError("I1 violation: fetch_all_chunks called")):
            scoped_bm25(be, Scope(doc_ids=(doc_a,), owner_id="alice"), "damages")


def test_scoped_hybrid_returns_only_in_scope_chunks():
    from baselines.retrieval import scoped_hybrid
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        doc_a = _ingest(be, "Section 74 of the Indian Contract Act, 1872 governs damages.")
        _ingest(be, "The Arbitration and Conciliation Act, 1996 applies here.")

        result = scoped_hybrid(be, Scope(doc_ids=(doc_a,), owner_id="alice"), "damages",
                               _FakeEncoder(), _FakeReranker())
        assert result
        assert all(rc.chunk.doc_id == doc_a for rc in result)


def test_scoped_hybrid_empty_for_nonmatching_scope():
    from baselines.retrieval import scoped_hybrid
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        assert scoped_hybrid(be, Scope(doc_ids=("doc-nonexistent",), owner_id="alice"),
                             "anything", _FakeEncoder(), _FakeReranker()) == []


def test_scoped_hybrid_never_calls_fetch_all_chunks():
    from baselines.retrieval import scoped_hybrid
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        doc_a = _ingest(be, "Section 74 of the Indian Contract Act, 1872 governs damages.")
        with mock.patch.object(type(be), "fetch_all_chunks",
                               side_effect=AssertionError("I1 violation: fetch_all_chunks called")):
            scoped_hybrid(be, Scope(doc_ids=(doc_a,), owner_id="alice"), "damages",
                         _FakeEncoder(), _FakeReranker())
