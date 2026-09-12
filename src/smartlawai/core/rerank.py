"""Cross-encoder re-ranker: re-scores fused candidates to a precise top-k.
Conforms to protocols.Reranker (PLAN.md M2) with device awareness (CLAUDE.md #2)."""
from __future__ import annotations

from smartlawai import config
from smartlawai.adapters.base import Chunk, RetrievedChunk


class Reranker:
    """Cross-encoder reranker wrapping ms-marco-MiniLM-L-6-v2."""

    def __init__(
        self,
        device: str | None = None,
        model_id: str = config.RERANKER_MODEL_ID,
    ) -> None:
        from sentence_transformers import CrossEncoder

        from train.common.device import resolve_device

        self.device = str(resolve_device(device))
        self.model_id = model_id
        self.model = CrossEncoder(model_id, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: list[Chunk] | list[tuple[str, str]],
        top_k: int = config.RERANK_TOP_K,
    ) -> list[RetrievedChunk] | list[tuple[str, float]]:
        """Scores candidate chunks or (id, text) tuples against the query.
        When passed list[Chunk], returns list[RetrievedChunk] per protocols.Reranker.
        When passed list[tuple[str, str]], returns list[tuple[str, float]] for backwards compatibility."""
        if not candidates:
            return []

        first = candidates[0]
        if isinstance(first, Chunk):
            chunk_list: list[Chunk] = candidates  # type: ignore[assignment]
            pairs = [(query, c.chunk_text) for c in chunk_list]
            scores = self.model.predict(pairs)
            ranked = sorted(zip(chunk_list, scores), key=lambda x: x[1], reverse=True)
            return [RetrievedChunk(chunk=c, score=float(s)) for c, s in ranked[:top_k]]
        else:
            tuple_list: list[tuple[str, str]] = candidates  # type: ignore[assignment]
            pairs = [(query, p[1]) for p in tuple_list]
            scores = self.model.predict(pairs)
            ranked_tuples = sorted(
                zip([p[0] for p in tuple_list], scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return [(cid, float(s)) for cid, s in ranked_tuples[:top_k]]
