"""BM25 sparse retriever for exact statutory term matching (e.g. 'Section 138').
Fused with dense FAISS via Reciprocal Rank Fusion in rag.py."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunk_ids: list[str], texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi
        self.chunk_ids = chunk_ids
        self.bm25 = BM25Okapi([_tok(t) for t in texts]) if texts else None

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tok(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]
