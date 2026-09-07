"""I1-respecting retrieval for M7 baselines: every function here takes a Scope
and reaches the backend only through fetch_chunks_scoped -- never
fetch_all_chunks(). Built fresh rather than reusing core/rag.py::AdvisoryRAG,
which calls fetch_all_chunks() directly (an I1 violation) and depends on
FaithfulnessEvaluator, which fabricates a 0.5 default on failure (the exact I2
bug CLAUDE.md's own example warns against). A milestone whose purpose is
validating the harness cleanly should not route through code with two live
invariant violations."""
from __future__ import annotations

from smartlawai.adapters.base import BackendInterface, RetrievedChunk
from smartlawai.core.bm25 import BM25Retriever
from smartlawai.scope import Scope

from . import config

_RRF_K = 60  # matches core/rag.py's RRF_K -- same fusion constant, no reason to diverge


def _rrf(ranked_lists: list[list[str]]) -> dict[str, float]:
    """Small local copy of core/rag.py::_rrf's logic. Not imported: that
    function is underscore-prefixed (private to its own module), so importing
    it across module boundaries would couple baselines/ to another module's
    implementation detail. Six lines duplicated is cheaper than that coupling."""
    fused: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            if cid:
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
    return fused


def scoped_bm25(backend: BackendInterface, scope: Scope, query: str,
                 k: int = config.BM25_TOP_K) -> list[RetrievedChunk]:
    """BM25-only retrieval, scoped. This is the deliberate sparse-only
    condition for the 'BM25 + frontier model' baseline (RESEARCH.md 4.2) --
    no dense, no rerank, no HyDE."""
    chunks = backend.fetch_chunks_scoped(scope)  # I1: the only retrieval entry point
    if not chunks:
        return []
    by_id = {c.chunk_id: c for c in chunks}
    bm25 = BM25Retriever(list(by_id.keys()), [c.chunk_text for c in chunks])
    return [RetrievedChunk(chunk=by_id[cid], score=float(score))
            for cid, score in bm25.search(query, k=k) if cid in by_id]


def scoped_hybrid(backend: BackendInterface, scope: Scope, query: str,
                   encoder, reranker,
                   k_retrieve: int = config.RAG_RETRIEVE_TOP_K,
                   k_rerank: int = config.RAG_RERANK_TOP_K) -> list[RetrievedChunk]:
    """BM25 + dense (InLegalBERT) fused via RRF, then cross-encoder reranked.
    Shared by every '+RAG' baseline condition (Mistral, SaulLM, frontier) so
    they're all compared against the same retrieval quality -- only the
    generator differs between them."""
    chunks = backend.fetch_chunks_scoped(scope)  # I1, same entry point
    if not chunks:
        return []
    by_id = {c.chunk_id: c for c in chunks}
    ids = list(by_id.keys())
    texts = [c.chunk_text for c in chunks]

    bm25 = BM25Retriever(ids, texts)
    sparse_ids = [cid for cid, _ in bm25.search(query, k=k_retrieve)]

    vecs = encoder.embed(texts)
    qv = encoder.embed([query])
    dense_scores = vecs @ qv[0]  # both L2-normalised (inlegalbert.py) -> cosine
    order = dense_scores.argsort()[::-1][:k_retrieve]
    dense_ids = [ids[i] for i in order]

    fused = _rrf([dense_ids, sparse_ids])
    cand_ids = [cid for cid, _ in
                sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k_retrieve]]

    reranked = reranker.rerank(query, [(cid, by_id[cid].chunk_text) for cid in cand_ids],
                               k_rerank)
    return [RetrievedChunk(chunk=by_id[cid], score=float(score))
            for cid, score in reranked if cid in by_id]
