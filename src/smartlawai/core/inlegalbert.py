"""InLegalBERT encoder (law-ai/InLegalBERT): 768-d sentence embeddings via
mean-pooling + L2 normalisation (dot-product == cosine, matches FAISS IndexFlatIP).
torch/transformers imported lazily."""
from __future__ import annotations

import numpy as np

MODEL_ID = "law-ai/InLegalBERT"
DIM = 768


class InLegalBERTEncoder:
    def __init__(self, device: str | None = None, max_len: int = 512) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device).eval()

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.max_len,
                               return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(1)
                counts = mask.sum(1).clamp(min=1e-9)
                mean = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
                out.append(mean.cpu().numpy().astype("float32"))
        return np.vstack(out) if out else np.zeros((0, DIM), dtype="float32")
