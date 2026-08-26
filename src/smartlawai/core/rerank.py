"""Cross-encoder re-ranker: re-scores top-20 fused candidates to a precise top-5."""
from __future__ import annotations

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(RERANK_MODEL)

    def rerank(self, query: str, passages: list[tuple[str, str]],
               top_k: int = 5) -> list[tuple[str, float]]:
        if not passages:
            return []
        scores = self.model.predict([(query, p[1]) for p in passages])
        ranked = sorted(zip([p[0] for p in passages], scores),
                        key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:top_k]]
