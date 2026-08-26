"""Tests for expanded evaluation metrics (Phase 2)."""
from __future__ import annotations

from smartlawai.eval.metrics import (
    citation_precision,
    citation_recall,
    mean_reciprocal_rank,
    ner_f1,
    ner_per_label_f1,
    prf,
    precision_at_k,
    recall_at_k,
    rouge_l,
    token_f1,
)


# --------------------------------------------------------------------------- #
# Original metrics
# --------------------------------------------------------------------------- #
def test_rouge_l_exact():
    assert rouge_l("a b c", "a b c") == 1.0

def test_rouge_l_partial():
    score = rouge_l("a b c d", "a b e f")
    assert 0 < score < 1.0

def test_rouge_l_empty():
    assert rouge_l("", "a b c") == 0.0

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
