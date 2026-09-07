"""TREC pooling tests (M6 Phase 10) -- RESEARCH.md §7.1.2."""
from __future__ import annotations

import pytest

from eval import config
from eval.annotate.pooling import (
    pool_candidates,
    pool_stats,
    unique_contribution,
)

RUNS = {
    "bm25": ["c1", "c2", "c3", "c4", "c5"],
    "dense": ["c3", "c6", "c1", "c7", "c8"],
    "hybrid": ["c1", "c3", "c9", "c2", "c10"],
}


def test_pool_deduplicates_across_methods():
    ids = [c.chunk_id for c in pool_candidates(RUNS)]
    assert len(ids) == len(set(ids))
    assert set(ids) == {f"c{i}" for i in range(1, 11)}


def test_pool_records_which_methods_found_each_candidate():
    by_id = {c.chunk_id: c for c in pool_candidates(RUNS)}
    assert by_id["c1"].found_by == ["bm25", "dense", "hybrid"]
    assert by_id["c6"].found_by == ["dense"]


def test_pool_ranks_multiply_retrieved_candidates_first():
    """A chunk found near the top by several retrievers outranks a single hit."""
    pool = pool_candidates(RUNS)
    ranks = {c.chunk_id: c.pooled_rank for c in pool}
    assert ranks["c1"] < ranks["c6"]
    assert ranks["c3"] < ranks["c8"]


def test_pool_ranks_are_contiguous_from_one():
    pool = pool_candidates(RUNS)
    assert [c.pooled_rank for c in pool] == list(range(1, len(pool) + 1))


def test_pool_depth_is_respected():
    pool = pool_candidates(RUNS, depth=2)
    assert {c.chunk_id for c in pool} == {"c1", "c2", "c3", "c6"}


def test_default_depth_is_top_ten_per_method():
    assert config.POOL_DEPTH == 10
    assert config.POOL_METHODS == ("bm25", "dense", "hybrid")


def test_pool_can_be_capped_to_the_target_judgment_budget():
    """RESEARCH.md §7.1.2: judge only ~15 unique candidates per query."""
    pool = pool_candidates(RUNS, max_candidates=5)
    assert len(pool) == 5
    assert pool[0].pooled_rank == 1


def test_pool_order_is_deterministic_across_runs_and_input_order():
    """Two raters must see the same order, on any machine."""
    reordered = {k: RUNS[k] for k in reversed(list(RUNS))}
    assert ([c.chunk_id for c in pool_candidates(RUNS)]
            == [c.chunk_id for c in pool_candidates(reordered)])


def test_pool_is_not_grouped_by_contributing_method():
    """Grouping would let a rater infer which system found a candidate,
    reintroducing exactly the bias blinding removes."""
    pool = pool_candidates(RUNS)
    single_method_positions = [i for i, c in enumerate(pool)
                               if c.found_by == ["dense"]]
    assert single_method_positions != list(range(len(single_method_positions)))


def test_pool_records_the_best_rank_seen():
    by_id = {c.chunk_id: c for c in pool_candidates(RUNS)}
    assert by_id["c1"].best_rank == 1
    assert by_id["c2"].best_rank == 2


def test_pool_of_a_single_method_preserves_its_order():
    pool = pool_candidates({"bm25": ["a", "b", "c"]})
    assert [c.chunk_id for c in pool] == ["a", "b", "c"]


def test_pool_of_empty_runs_is_empty():
    assert pool_candidates({}) == []
    assert pool_candidates({"bm25": []}) == []


def test_pool_rejects_a_non_positive_depth():
    with pytest.raises(ValueError, match="positive"):
        pool_candidates(RUNS, depth=0)


# --------------------------------------------------------------------------- #
# Budget arithmetic -- the justification for pooling at all
# --------------------------------------------------------------------------- #
def test_pool_stats_computes_the_judgment_budget():
    pools = {f"q{i}": pool_candidates(RUNS, max_candidates=15) for i in range(200)}
    stats = pool_stats(pools)
    assert stats["n_queries"] == 200
    assert stats["n_judgments"] == 200 * 10
    assert stats["mean_pool_size"] == 10.0
    # The RESEARCH.md §7.1.2 claim: ~3,000 judgments for 200 queries, not millions.
    assert stats["n_judgments"] < 3500


def test_pool_stats_reports_per_method_contribution():
    stats = pool_stats({"q1": pool_candidates(RUNS)})
    assert set(stats["contribution_by_method"]) == {"bm25", "dense", "hybrid"}
    assert all(v == 5 for v in stats["contribution_by_method"].values())


def test_pool_stats_of_nothing_is_zero_not_an_error():
    stats = pool_stats({})
    assert stats["n_queries"] == 0 and stats["mean_pool_size"] is None


def test_unique_contribution_identifies_a_redundant_retriever():
    runs = {"bm25": ["a", "b", "c"], "clone": ["a", "b", "c"],
            "dense": ["x", "y", "z"]}
    contribution = unique_contribution(runs)
    assert contribution["bm25"] == 0
    assert contribution["clone"] == 0
    assert contribution["dense"] == 3
