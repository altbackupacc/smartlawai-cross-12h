"""TREC pooling for the retrieval gold set.

RESEARCH.md §7.1.2 / M6_ONBOARDING.md §7: judging every chunk against every
query is impossible. Run BM25, dense and hybrid retrieval, pool each method's
top-10, and judge only the ~15 unique candidates per query -- which reduces 200
queries from millions of judgments to roughly 3,000.

The pooling logic and the queries are what M6 owns. Which retriever produces the
runs is not: today they are the M0 stubs, after M2 they are real, and this module
takes runs as input either way so nothing here changes when they are swapped
(M6_ONBOARDING.md §7: "the queries and pooling logic are what you're building;
swap in real retrieval once M2 lands").

Two properties this module guarantees, because both are what makes the resulting
judgments usable in a paper:

  * **Fairness.** A candidate is judged on its merits, not on which system found
    it. The pooled list is ordered deterministically by pooled rank and then by
    id -- never grouped by contributing method, which would let a rater infer
    the source and reintroduce the bias blinding removes.
  * **Recorded provenance.** Every candidate remembers which methods retrieved
    it, so a later "which retriever contributed unjudged relevant documents"
    analysis is possible without re-running anything.

Imports nothing from `smartlawai` (OPS.md §8).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from eval import config


@dataclass
class PooledCandidate:
    """One chunk in a query's judgment pool."""

    chunk_id: str
    pooled_rank: int                                  # 1 = best reciprocal-rank score
    found_by: list[str] = field(default_factory=list)  # contributing methods
    best_rank: int = 0                                 # best rank across methods
    rrf_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "pooled_rank": self.pooled_rank,
                "found_by": self.found_by, "best_rank": self.best_rank,
                "rrf_score": round(self.rrf_score, 6)}


# Reciprocal-rank-fusion constant. 60 is the value from Cormack et al. (2009),
# used unchanged so the pooling order is a citable choice rather than a tuned one.
RRF_K: int = 60


def pool_candidates(runs: Mapping[str, Sequence[str]],
                    depth: int = config.POOL_DEPTH,
                    max_candidates: int | None = None) -> list[PooledCandidate]:
    """Pool the top-`depth` results of each retrieval run into one judgment list.

    runs: {method_name: ranked chunk_ids}. Any set of methods works; the three
    RESEARCH.md §7.1.2 names are the default (`config.POOL_METHODS`).

    Ordering is reciprocal rank fusion across methods, so a chunk found near the
    top by two retrievers precedes one found once. Ties break on chunk_id, so
    the pool is byte-identical across machines and reruns -- a rater re-opening
    the app sees the same order, and two raters see the same order as each other.

    `max_candidates` caps the pool (RESEARCH.md's ~15 per query). Truncation
    keeps the highest-fused candidates, which is the standard depth-k pool.
    """
    if depth <= 0:
        raise ValueError("pool depth must be positive")

    scores: dict[str, float] = {}
    found_by: dict[str, list[str]] = {}
    best_rank: dict[str, int] = {}

    for method in sorted(runs):
        for rank, chunk_id in enumerate(list(runs[method])[:depth], start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            found_by.setdefault(chunk_id, []).append(method)
            if chunk_id not in best_rank or rank < best_rank[chunk_id]:
                best_rank[chunk_id] = rank

    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
    if max_candidates is not None:
        ordered = ordered[:max_candidates]

    return [PooledCandidate(chunk_id=cid, pooled_rank=i,
                            found_by=sorted(found_by[cid]),
                            best_rank=best_rank[cid], rrf_score=scores[cid])
            for i, cid in enumerate(ordered, start=1)]


def pool_stats(pools: Mapping[str, Sequence[PooledCandidate]]) -> dict[str, Any]:
    """Judgment-budget arithmetic for a whole query set.

    This is the number RESEARCH.md §7.1.2 uses to justify pooling at all
    ("reduces 200 queries from millions of judgments to ~3,000"), so it is
    computed from the actual pools rather than quoted.
    """
    sizes = [len(p) for p in pools.values()]
    if not sizes:
        return {"n_queries": 0, "n_judgments": 0, "mean_pool_size": None,
                "min_pool_size": None, "max_pool_size": None,
                "unique_chunks": 0, "contribution_by_method": {}}

    by_method: dict[str, int] = {}
    unique: set[str] = set()
    for pool in pools.values():
        for candidate in pool:
            unique.add(candidate.chunk_id)
            for method in candidate.found_by:
                by_method[method] = by_method.get(method, 0) + 1

    return {"n_queries": len(sizes),
            "n_judgments": sum(sizes),
            "mean_pool_size": round(sum(sizes) / len(sizes), 2),
            "min_pool_size": min(sizes),
            "max_pool_size": max(sizes),
            "unique_chunks": len(unique),
            "contribution_by_method": dict(sorted(by_method.items()))}


def unique_contribution(runs: Mapping[str, Sequence[str]],
                        depth: int = config.POOL_DEPTH) -> dict[str, int]:
    """How many pooled candidates each method alone contributed.

    A method contributing nothing unique is a method the pool does not need --
    worth knowing before paying for a third retriever's judgments.
    """
    seen: dict[str, set[str]] = {m: set(list(runs[m])[:depth]) for m in runs}
    out: dict[str, int] = {}
    for method, chunks in seen.items():
        others: set[str] = set()
        for other, ids in seen.items():
            if other != method:
                others |= ids
        out[method] = len(chunks - others)
    return dict(sorted(out.items()))
