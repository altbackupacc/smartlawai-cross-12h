"""Unit tests for core/rerank.py (PLAN.md M2)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from smartlawai.adapters.base import Chunk, RetrievedChunk
from smartlawai.core.rerank import Reranker
from smartlawai.protocols import Reranker as RerankerProtocol


def test_reranker_protocol_conformance_with_chunks():
    """Verify Reranker implements protocols.Reranker and returns RetrievedChunk."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.2, 0.9, 0.5], dtype="float32")

    with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
        reranker = Reranker(device="cpu")
        assert isinstance(reranker, RerankerProtocol)

        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", chunk_index=0, chunk_text="low match",
                  char_start=0, char_end=9),
            Chunk(chunk_id="c2", doc_id="d1", chunk_index=1, chunk_text="high match",
                  char_start=10, char_end=20),
            Chunk(chunk_id="c3", doc_id="d1", chunk_index=2, chunk_text="mid match",
                  char_start=21, char_end=30),
        ]

        results = reranker.rerank("query text", chunks, top_k=2)
        assert len(results) == 2
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].chunk.chunk_id == "c2"
        assert np.isclose(results[0].score, 0.9, atol=1e-4)
        assert results[1].chunk.chunk_id == "c3"
        assert np.isclose(results[1].score, 0.5, atol=1e-4)


def test_reranker_backward_compatible_with_tuples():
    """Verify Reranker continues to support (id, text) tuple inputs."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.1, 0.8], dtype="float32")

    with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
        reranker = Reranker(device="cpu")
        passages = [("id1", "passage 1"), ("id2", "passage 2")]
        results = reranker.rerank("query text", passages, top_k=5)

        assert len(results) == 2
        assert results[0][0] == "id2"
        assert np.isclose(results[0][1], 0.8, atol=1e-5)
        assert results[1][0] == "id1"
        assert np.isclose(results[1][1], 0.1, atol=1e-5)


def test_reranker_empty():
    with patch("sentence_transformers.CrossEncoder"):
        reranker = Reranker(device="cpu")
        assert reranker.rerank("query", []) == []
