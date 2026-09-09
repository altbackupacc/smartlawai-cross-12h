"""InLegalBERT encoder: 768-d sentence embeddings via mean-pooling + L2 normalisation
(dot-product == cosine, matches FAISS IndexFlatIP). Lazy imports and device-aware (CLAUDE.md #2)."""
from __future__ import annotations

from typing import Any

import numpy as np

from smartlawai import config


def mean_pool_and_normalize(hidden_state: Any, attention_mask: Any) -> Any:
    """Computes mean pooling over unmasked token embeddings and applies L2 normalisation.
    Matches IndexFlatIP so dot-product equals cosine similarity."""
    import torch

    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_state * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(summed / counts, p=2, dim=1)


class InLegalBERTEncoder:
    """Encodes legal text into 768-dimensional L2-normalised vectors."""

    def __init__(
        self,
        device: str | None = None,
        max_len: int = config.INLEGALBERT_MAX_LEN,
        model_id: str = config.INLEGALBERT_MODEL_ID,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from train.common.device import resolve_device

        self._torch = torch
        self.device = resolve_device(device)
        self.max_len = max_len
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Embeds a list of texts into L2-normalised float32 vectors."""
        if not texts:
            return np.zeros((0, config.INLEGALBERT_DIM), dtype="float32")

        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hidden = self.model(**enc).last_hidden_state
                normed = mean_pool_and_normalize(hidden, enc["attention_mask"])
                out.append(normed.cpu().float().numpy().astype("float32"))
        return np.vstack(out) if out else np.zeros((0, config.INLEGALBERT_DIM), dtype="float32")
