"""Tests for the M6 evaluation metrics.

Originally written against src/smartlawai/eval/metrics.py; the module moved to
the top-level eval/ package PLAN.md specifies (M6_ONBOARDING.md §4) and the
hand-rolled rouge_l was replaced by the rouge-score-backed rouge_scores. The
exact-match / partial-match / empty-string assertions are kept -- they are still
valid, just aimed at the new function.
"""
from __future__ import annotations

import pytest

from eval.metrics import (
    MetricResult,
    citation_coverage,
    citation_precision,
    citation_recall,
    citation_support_rate,
    citation_validity_rate,
    groundedness_rate,
    hallucination_rate,
    latency_summary,
    mean_reciprocal_rank,
    ndcg_at_k,
    ner_f1,
    ner_per_label_f1,
    per_perturbation_detection_rate,
    percentile,
    precision_at_k,
    prf,
    recall_at_k,
    refusal_metrics,
    registry_coverage_rate,
    repealed_citation_rate,
    retrieval_metrics,
    rouge_scores,
    token_f1,
)


# --------------------------------------------------------------------------- #
# Original metrics
# --------------------------------------------------------------------------- #
def test_rouge_exact():
    scores = rouge_scores("a b c", "a b c")
    assert scores["rouge1"] == 1.0
    assert scores["rougeL"] == 1.0

def test_rouge_partial():
    score = rouge_scores("a b c d", "a b e f")["rougeL"]
    assert 0 < score < 1.0

def test_rouge_empty():
    assert rouge_scores("", "a b c")["rougeL"] == 0.0

def test_rouge_reports_all_three_variants():
    """RESEARCH.md §5 requires ROUGE-1/2/L, not just L."""
    assert set(rouge_scores("the cat sat", "the cat sat")) == {"rouge1", "rouge2", "rougeL"}

def test_bertscore_is_present_but_unexercised():
    """bert-score pulls torch and has never been installed in this repo; no test
    has ever called bertscore_f1 (M6_ONBOARDING.md §18 flagged this as unknown --
    it is now checked). Skip honestly rather than pretend the metric works."""
    pytest.importorskip("bert_score",
                        reason="bert-score not installed; install the [eval] extra")
    from eval.metrics import bertscore_f1
    assert 0.0 <= bertscore_f1(["a legal claim"], ["a legal claim"]) <= 1.0

def test_token_f1_exact():
    assert token_f1("a b", "a b") == 1.0

def test_token_f1_partial():
    score = token_f1("a b c", "a b d")
    assert 0 < score < 1.0

def test_prf_perfect():
    assert prf(5, 0, 0) == (1.0, 1.0, 1.0)

def test_prf_zero():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# NER metrics
# --------------------------------------------------------------------------- #
def test_ner_f1_perfect():
    pred = [{"text": "IPC", "label": "STATUTE"}, {"text": "Delhi", "label": "LOCATION"}]
    gold = [{"text": "IPC", "label": "STATUTE"}, {"text": "Delhi", "label": "LOCATION"}]
    p, r, f = ner_f1(pred, gold)
    assert p == 1.0 and r == 1.0 and f == 1.0

def test_ner_f1_partial():
    pred = [{"text": "IPC", "label": "STATUTE"}, {"text": "CrPC", "label": "STATUTE"}]
    gold = [{"text": "IPC", "label": "STATUTE"}, {"text": "BNS", "label": "STATUTE"}]
    p, r, f = ner_f1(pred, gold)
    assert p == 0.5  # 1 correct out of 2 predicted
    assert r == 0.5  # 1 correct out of 2 gold
    assert f == 0.5

def test_ner_f1_empty():
    p, r, f = ner_f1([], [])
    assert p == 0.0 and r == 0.0 and f == 0.0

def test_ner_per_label():
    pred = [{"text": "IPC", "label": "STATUTE"}, {"text": "Delhi", "label": "LOCATION"}]
    gold = [{"text": "IPC", "label": "STATUTE"}, {"text": "Mumbai", "label": "LOCATION"}]
    results = ner_per_label_f1(pred, gold)
    assert "STATUTE" in results
    assert "LOCATION" in results
    assert results["STATUTE"][2] == 1.0  # F1 for STATUTE is perfect


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def test_mrr_first_hit():
    rankings = [["a", "b", "c"]]
    relevants = [{"a"}]
    assert mean_reciprocal_rank(rankings, relevants) == 1.0

def test_mrr_second_hit():
    rankings = [["a", "b", "c"]]
    relevants = [{"b"}]
    assert mean_reciprocal_rank(rankings, relevants) == 0.5

def test_mrr_no_hit():
    rankings = [["a", "b", "c"]]
    relevants = [{"d"}]
    assert mean_reciprocal_rank(rankings, relevants) == 0.0

def test_recall_at_k():
    rankings = [["a", "b", "c", "d", "e"]]
    relevants = [{"a", "c", "f"}]
    # top-5 has a and c → 2/3 relevant found
    r5 = recall_at_k(rankings, relevants, k=5)
    assert abs(r5 - 0.6667) < 0.001

def test_precision_at_k():
    rankings = [["a", "b", "c", "d", "e"]]
    relevants = [{"a", "c"}]
    # 2 of 5 are relevant → 0.4
    p5 = precision_at_k(rankings, relevants, k=5)
    assert abs(p5 - 0.4) < 0.001


# --------------------------------------------------------------------------- #
# Citation metrics
# --------------------------------------------------------------------------- #
def test_citation_recall_perfect():
    pred = [{"a", "b"}]
    gold = [{"a", "b"}]
    assert citation_recall(pred, gold) == 1.0

def test_citation_recall_partial():
    pred = [{"a"}]
    gold = [{"a", "b"}]
    assert citation_recall(pred, gold) == 0.5

def test_citation_precision_partial():
    pred = [{"a", "b", "c"}]
    gold = [{"a"}]
    p = citation_precision(pred, gold)
    assert abs(p - 0.3333) < 0.001


# --------------------------------------------------------------------------- #
# nDCG (RESEARCH.md §5: nDCG@10)
# --------------------------------------------------------------------------- #
def test_ndcg_perfect_ranking_is_one():
    rankings = [["a", "b", "c"]]
    relevances = [{"a": 3.0, "b": 2.0, "c": 1.0}]
    assert ndcg_at_k(rankings, relevances, k=3) == 1.0

def test_ndcg_reversed_ranking_matches_hand_computation():
    # gains in rank order: 1, 2, 3 -> DCG = 1/log2(2) + 2/log2(3) + 3/log2(4)
    #                                    = 1.0 + 1.26186 + 1.5 = 3.76186
    # ideal order 3, 2, 1          -> IDCG = 3.0 + 1.26186 + 0.5 = 4.76186
    rankings = [["c", "b", "a"]]
    relevances = [{"a": 3.0, "b": 2.0, "c": 1.0}]
    expected = 3.761859507 / 4.761859507
    assert abs(ndcg_at_k(rankings, relevances, k=3) - round(expected, 4)) < 0.001

def test_ndcg_binary_relevance_is_the_all_ones_special_case():
    rankings = [["x", "a", "b"]]
    relevances = [{"a": 1.0, "b": 1.0}]
    score = ndcg_at_k(rankings, relevances, k=3)
    assert 0 < score < 1.0

def test_ndcg_skips_queries_with_no_judged_relevant_chunk():
    # The unjudged query must not drag the mean down with an unearned 0.0.
    rankings = [["a"], ["z"]]
    relevances = [{"a": 1.0}, {}]
    assert ndcg_at_k(rankings, relevances, k=3) == 1.0

def test_ndcg_empty_input():
    assert ndcg_at_k([], []) == 0.0


def test_retrieval_metrics_reports_the_full_research_row():
    rankings = [["a", "b", "c"]]
    relevances = [{"a": 1.0}]
    names = {m.name for m in retrieval_metrics(rankings, relevances)}
    assert {"recall@5", "recall@10", "recall@20", "mrr", "ndcg@10"} <= names

def test_retrieval_metrics_unavailable_not_zero_when_nothing_is_judged():
    results = retrieval_metrics([["a", "b"]], [{}])
    assert all(m.status == "unavailable" for m in results)
    assert all(m.value is None for m in results)
    assert all(m.reason == "no_relevance_judgments" for m in results)

def test_retrieval_metrics_unavailable_when_no_queries():
    results = retrieval_metrics([], [])
    assert all(m.status == "unavailable" and m.value is None for m in results)


# --------------------------------------------------------------------------- #
# Groundedness / hallucination (CLAUDE.md §5 -- one definition only)
# --------------------------------------------------------------------------- #
def test_groundedness_rate_all_supported():
    r = groundedness_rate([True, True, True])
    assert r.value == 1.0 and r.status == "ok" and r.n == 3

def test_groundedness_rate_mixed():
    assert groundedness_rate([True, False, True, False]).value == 0.5

def test_groundedness_excludes_unavailable_claims_from_the_denominator():
    # I2: an unavailable verifier is not evidence of a hallucination.
    r = groundedness_rate([True, None, None])
    assert r.value == 1.0
    assert r.n == 1
    assert r.detail["n_unavailable"] == 2

def test_groundedness_all_unavailable_is_unavailable_not_zero():
    r = groundedness_rate([None, None])
    assert r.value is None and r.status == "unavailable"

def test_hallucination_rate_is_exactly_one_minus_groundedness():
    """CLAUDE.md §5: one definition only. Any drift here is a bug."""
    for labels in ([True, False], [True, True, False, None], [False, False]):
        g = groundedness_rate(labels)
        h = hallucination_rate(labels)
        assert abs((g.value + h.value) - 1.0) < 1e-9

def test_hallucination_rate_propagates_unavailability():
    assert hallucination_rate([None]).status == "unavailable"


# --------------------------------------------------------------------------- #
# Citation metrics
# --------------------------------------------------------------------------- #
def test_citation_coverage():
    assert citation_coverage(3, 4).value == 0.75

def test_citation_coverage_no_claims_is_unavailable():
    assert citation_coverage(0, 0).status == "unavailable"

# Registry-result fixtures use M4's OWN published RegistryResult contract
# (M4_ONBOARDING.md §6: {status, as_of, note}) rather than an invented shape --
# corrected during the audit (docs/M6_GAP_ANALYSIS.md G2) so a real M4 result
# drops in with no translation layer the day that module lands.
def _res(status, note=None):
    return {"citation": "x", "status": status, "as_of": "2026-01-01", "note": note}

def test_registry_coverage_rate_mixed():
    r = registry_coverage_rate([_res("in_force"), _res("not_found")])
    assert r.value == 0.5 and r.detail["n_not_found"] == 1

def test_registry_coverage_rate_empty_is_unavailable():
    assert registry_coverage_rate([]).status == "unavailable"

def test_citation_validity_rate_counts_repealed_and_superseded_as_valid():
    # "Valid" = resolves to something real; repealed/superseded still exist,
    # only not_found means the citation doesn't (or isn't covered).
    r = citation_validity_rate([_res("in_force"), _res("repealed"),
                                _res("superseded"), _res("not_found")])
    assert r.value == 0.75 and r.n == 4

def test_citation_validity_excludes_nothing_from_the_denominator():
    # Unlike the pre-audit design, not_found citations count in the
    # denominator (they ARE the "invalid" cases) -- registry_coverage_rate is
    # the separate, explicitly-named metric for "how much did we even cover".
    r = citation_validity_rate([_res("in_force"), _res("not_found")])
    assert r.value == 0.5
    assert r.detail["n_not_found"] == 1

def test_citation_validity_empty_is_unavailable_not_zero():
    assert citation_validity_rate([]).status == "unavailable"

def test_citation_validity_rejects_a_malformed_status():
    r = citation_validity_rate([_res("maybe")])
    assert r.status == "unavailable" and "malformed_registry_status" in r.reason

def test_repealed_citation_rate():
    r = repealed_citation_rate([_res("repealed", note="see BNS s.318"),
                                _res("in_force")])
    assert r.value == 0.5
    assert r.detail["n_with_note"] == 1

def test_repealed_rate_treats_superseded_as_repealed_too():
    r = repealed_citation_rate([_res("superseded"), _res("in_force")])
    assert r.value == 0.5

def test_repealed_rate_does_not_count_not_found_as_in_force():
    # Counting not_found as in-force would understate the repealed rate -- the
    # exact direction of error that would flatter this project's own result.
    r = repealed_citation_rate([_res("repealed"), _res("not_found")])
    assert r.value == 1.0
    assert r.detail["n_not_found"] == 1

def test_repealed_rate_all_not_found_is_unavailable():
    r = repealed_citation_rate([_res("not_found")])
    assert r.status == "unavailable" and r.reason == "registry_resolved_nothing"

def test_repealed_rate_empty_is_unavailable():
    assert repealed_citation_rate([]).status == "unavailable"

def test_citation_support_rate_ignores_unjudged():
    r = citation_support_rate([True, False, None])
    assert r.value == 0.5 and r.detail["n_unjudged"] == 1

def test_citation_support_rate_all_unjudged_is_unavailable():
    assert citation_support_rate([None, None]).status == "unavailable"


# --------------------------------------------------------------------------- #
# Refusal / OOS
# --------------------------------------------------------------------------- #
def test_refusal_metrics_perfect_system():
    gold = ["in_scope", "out_of_scope", "advice_seeking"]
    decisions = ["ANSWER", "REFUSE", "REFUSE"]
    by_name = {m.name: m for m in refusal_metrics(gold, decisions)}
    assert by_name["refusal_precision"].value == 1.0
    assert by_name["refusal_recall"].value == 1.0
    assert by_name["refusal_f1"].value == 1.0
    assert by_name["over_refusal_rate"].value == 0.0
    assert by_name["refusal_accuracy"].value == 1.0

def test_refusal_metrics_always_refusing_system():
    gold = ["in_scope", "in_scope", "out_of_scope"]
    decisions = ["REFUSE", "REFUSE", "REFUSE"]
    by_name = {m.name: m for m in refusal_metrics(gold, decisions)}
    assert by_name["refusal_recall"].value == 1.0
    assert abs(by_name["refusal_precision"].value - 0.3333) < 0.001
    assert by_name["over_refusal_rate"].value == 1.0

def test_refusal_metrics_advice_seeking_counts_as_should_refuse():
    by_name = {m.name: m for m in refusal_metrics(["advice_seeking"], ["ANSWER"])}
    assert by_name["refusal_recall"].value == 0.0

def test_refusal_metrics_misaligned_input_is_unavailable():
    results = refusal_metrics(["in_scope"], [])
    assert all(m.status == "unavailable" for m in results)


# --------------------------------------------------------------------------- #
# Judge diagnostics
# --------------------------------------------------------------------------- #
def test_per_perturbation_detection_rate_splits_by_type():
    types = ["ipc_bns_swap", "ipc_bns_swap", "negate_obligation"]
    detected = [True, False, True]
    out = per_perturbation_detection_rate(types, detected)
    assert out["ipc_bns_swap"].value == 0.5
    assert out["negate_obligation"].value == 1.0

def test_per_perturbation_unjudged_type_is_unavailable_not_zero():
    out = per_perturbation_detection_rate(["change_court"], [None])
    assert out["change_court"].status == "unavailable"


# --------------------------------------------------------------------------- #
# Cost / latency
# --------------------------------------------------------------------------- #
def test_percentile_basic():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([], 0.5) is None

def test_latency_summary_per_stage():
    out = {m.name: m for m in latency_summary({"retrieve": [1.0, 2.0, 3.0]})}
    assert out["latency_p50_ms[retrieve]"].value == 2.0
    assert out["latency_p95_ms[retrieve]"].value is not None

def test_latency_summary_missing_stage_is_unavailable():
    out = {m.name: m for m in latency_summary({"verify": []})}
    assert out["latency_p50_ms[verify]"].status == "unavailable"


# --------------------------------------------------------------------------- #
# MetricResult contract (I2)
# --------------------------------------------------------------------------- #
def test_metric_result_unavailable_never_carries_a_number():
    m = MetricResult.unavailable("x", "because")
    assert m.value is None and m.status == "unavailable" and m.reason == "because"
    assert m.to_dict()["value"] is None

def test_metric_result_not_applicable_is_distinct_from_unavailable():
    assert MetricResult.not_applicable("x", "n/a").status == "not_applicable"
