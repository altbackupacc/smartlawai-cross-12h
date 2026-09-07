"""Smoke tests for the local backend (run: pytest tests/)."""
from __future__ import annotations

import os
import tempfile

import numpy as np


def _backend(tmp):
    os.environ["SMARTLAW_LOCAL_DIR"] = tmp
    from smartlawai.adapters.factory import get_backend
    return get_backend("local")


def test_document_and_chunks_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        from smartlawai.adapters.base import Document
        from smartlawai.core.chunking import chunk_document

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("1. First clause. WHEREAS the parties agree. 2. Second clause.")
            path = f.name
        doc = Document(doc_id="d1", filename="d1.txt", doc_type="CONTRACT",
                       language="en", source="test", storage_uri="",
                       raw_text="1. First clause. WHEREAS the parties agree. 2. Second.")
        be.upload_doc(path, doc)
        be.store_chunks(chunk_document("d1", doc.raw_text))

        assert "d1" in be.list_doc_ids()
        assert be.fetch_document_text("d1").startswith("1. First")
        assert len(be.fetch_chunks("d1")) >= 1


def test_faiss_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        idx = be.new_flat_ip_index(4)
        v = np.random.rand(2, 4).astype("float32")
        idx.add_with_ids(v, np.array([10, 20], dtype="int64"))
        be.save_faiss_index(idx, "t1")
        assert be.faiss_index_exists("t1")
        loaded = be.load_faiss_index("t1")
        _, ids = loaded.search(v[:1], 1)
        assert int(ids[0][0]) == 10


def test_metrics():
    # M6: metrics moved src/smartlawai/eval/metrics.py -> eval/metrics.py, and
    # the hand-rolled rouge_l was replaced by rouge_scores (PLAN.md M6).
    from eval.metrics import prf, rouge_scores, token_f1
    assert rouge_scores("a b c", "a b c")["rougeL"] == 1.0
    assert token_f1("a b", "a b") == 1.0
    assert prf(5, 0, 0) == (1.0, 1.0, 1.0)
