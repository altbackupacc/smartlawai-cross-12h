"""Unit tests for core/inlegalbert.py and core/encoders.py (PLAN.md M2)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from smartlawai.core.encoders import InLegalBERTEncoder, mean_pool_and_normalize
from smartlawai.protocols import Encoder


def test_mean_pool_and_normalize_mathematics():
    """Verify mean-pooling excludes masked tokens and produces exact unit norm vectors."""
    # Batch size 2, seq len 3, hidden dim 4
    hidden = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [99.0, 99.0, 99.0, 99.0]],  # token 2 is padding
        [[0.0, 0.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],       # tokens 1,2 padding
    ], dtype=torch.float32)

    attention_mask = torch.tensor([
        [1, 1, 0],
        [1, 0, 0],
    ], dtype=torch.long)

    normed = mean_pool_and_normalize(hidden, attention_mask)

    # First item average over tokens 0 and 1: [0.5, 1.0, 0.0, 0.0] -> norm sqrt(1.25)
    # Token 2 (value 99) must be ignored because attention_mask is 0.
    v0 = normed[0].numpy()
    assert np.isclose(np.linalg.norm(v0), 1.0, atol=1e-5)
    assert np.isclose(v0[0] / v0[1], 0.5, atol=1e-5)
    assert np.isclose(v0[2], 0.0)
    assert np.isclose(v0[3], 0.0)

    # Second item token 0: [0, 0, 3, 4] -> norm is 5, normalized is [0, 0, 0.6, 0.8]
    v1 = normed[1].numpy()
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)
    assert np.isclose(v1[2], 0.6, atol=1e-5)
    assert np.isclose(v1[3], 0.8, atol=1e-5)


def test_encoder_empty_input():
    """Embedding an empty list of texts returns an empty array of shape (0, 768)."""
    with patch("transformers.AutoTokenizer.from_pretrained"), patch("transformers.AutoModel.from_pretrained"):
        enc = InLegalBERTEncoder(device="cpu")
        res = enc.embed([])
        assert isinstance(res, np.ndarray)
        assert res.shape == (0, 768)


def test_encoder_mocked_batch_inference():
    """Verify embed processes batches and conforms to protocols.Encoder."""
    mock_tok = MagicMock()
    mock_model = MagicMock()

    # Mock tokenizer output
    mock_tok.return_value = {
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        "input_ids": torch.tensor([[101, 102], [101, 102]]),
    }
    # Mock model last hidden state: 2 texts, seq len 2, dim 768
    hidden_state = torch.ones((2, 2, 768), dtype=torch.float32)
    mock_model.return_value.last_hidden_state = hidden_state
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok), \
         patch("transformers.AutoModel.from_pretrained", return_value=mock_model):

        encoder = InLegalBERTEncoder(device="cpu")
        assert isinstance(encoder, Encoder)

        vectors = encoder.embed(["test clause 1", "test clause 2"], batch_size=2)
        assert vectors.shape == (2, 768)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
