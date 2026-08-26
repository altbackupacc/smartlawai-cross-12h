"""Metric functions: ROUGE-L, token-F1, precision/recall/F1, optional BERTScore,
NER F1, retrieval metrics (MRR, Recall@K), and citation coverage.

Expanded in Phase 2 to cover all 5 synopsis evaluation dimensions:
  1. Summarisation quality (ROUGE-L, token-F1, BERTScore)
  2. Clause extraction accuracy (P/R/F1)
  3. QA / RAG correctness (token-F1, Accuracy)
  4. Hallucination detection (hallucination rate)
  5. NER accuracy (P/R/F1)
  6. Retrieval quality (MRR, Recall@K)
  7. Citation coverage
"""
from __future__ import annotations

from collections import Counter


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L = F1 over the longest common subsequence (token level)."""
    p, r = pred.split(), ref.split()
    if not p or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            dp[i][j] = (dp[i - 1][j - 1] + 1 if p[i - 1] == r[j - 1]
                        else max(dp[i - 1][j], dp[i][j - 1]))
    lcs = dp[-1][-1]
    prec, rec = lcs / len(p), lcs / len(r)
    return 0.0 if prec + rec == 0 else round(2 * prec * rec / (prec + rec), 4)


def token_f1(pred: str, ref: str) -> float:
    """Token-level F1 between predicted and reference text."""
    pc, rc = Counter(pred.lower().split()), Counter(ref.lower().split())
    common = sum((pc & rc).values())
    if common == 0:
        return 0.0
    prec, rec = common / sum(pc.values()), common / sum(rc.values())
    return round(2 * prec * rec / (prec + rec), 4)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, Recall, F1 from raw counts."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def bertscore_f1(preds: list[str], refs: list[str]) -> float:
    """BERTScore F1 (requires bert-score package)."""
    from bert_score import score
    _, _, f1 = score(preds, refs, lang="en", rescale_with_baseline=True)
    return round(float(f1.mean()), 4)


# --------------------------------------------------------------------------- #
# NER evaluation
# --------------------------------------------------------------------------- #
def ner_f1(pred_entities: list[dict], gold_entities: list[dict],
           match_on: str = "text_label") -> tuple[float, float, float]:
    """Compute NER Precision/Recall/F1.

    Each entity is a dict with at least 'text' and 'label' keys.
    match_on: 'text_label' matches (text.lower(), label) pairs.
              'span' matches (start, end, label) tuples.

    Returns: (precision, recall, f1)
    """
    if match_on == "span":
        pred_set = {(e["start"], e["end"], e["label"]) for e in pred_entities}
        gold_set = {(e["start"], e["end"], e["label"]) for e in gold_entities}
    else:
        pred_set = {(e["text"].lower().strip(), e["label"]) for e in pred_entities}
        gold_set = {(e["text"].lower().strip(), e["label"]) for e in gold_entities}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return prf(tp, fp, fn)


def ner_per_label_f1(pred_entities: list[dict],
                     gold_entities: list[dict]) -> dict[str, tuple[float, float, float]]:
    """Compute per-label P/R/F1 for NER."""
    labels = {e["label"] for e in pred_entities} | {e["label"] for e in gold_entities}
    results = {}
    for label in sorted(labels):
        p_sub = [e for e in pred_entities if e["label"] == label]
        g_sub = [e for e in gold_entities if e["label"] == label]
        results[label] = ner_f1(p_sub, g_sub)
    return results


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #
def mean_reciprocal_rank(rankings: list[list[str]],
                         relevants: list[set[str]]) -> float:
    """Mean Reciprocal Rank (MRR) across queries.

    rankings: list of ranked doc/chunk IDs per query.
    relevants: set of relevant IDs per query.
    """
    if not rankings:
        return 0.0
    total = 0.0
    for ranked, rel in zip(rankings, relevants):
        for i, doc_id in enumerate(ranked, 1):
            if doc_id in rel:
                total += 1.0 / i
                break
    return round(total / len(rankings), 4)


def recall_at_k(rankings: list[list[str]], relevants: list[set[str]],
                k: int = 5) -> float:
    """Recall@K: fraction of relevant documents appearing in top-K results."""
    if not rankings:
        return 0.0
    scores = []
    for ranked, rel in zip(rankings, relevants):
        top_k = set(ranked[:k])
        if rel:
            scores.append(len(top_k & rel) / len(rel))
        else:
            scores.append(0.0)
    return round(sum(scores) / len(scores), 4)


def precision_at_k(rankings: list[list[str]], relevants: list[set[str]],
                   k: int = 5) -> float:
    """Precision@K: fraction of top-K results that are relevant."""
    if not rankings:
        return 0.0
    scores = []
    for ranked, rel in zip(rankings, relevants):
        top_k = ranked[:k]
        if top_k:
            scores.append(sum(1 for d in top_k if d in rel) / len(top_k))
        else:
            scores.append(0.0)
    return round(sum(scores) / len(scores), 4)


# --------------------------------------------------------------------------- #
# Citation coverage
# --------------------------------------------------------------------------- #
def citation_recall(pred_citations: list[set[str]],
                    gold_citations: list[set[str]]) -> float:
    """Average citation recall: what fraction of gold citations appear in predicted."""
    if not pred_citations:
        return 0.0
    scores = []
    for pred, gold in zip(pred_citations, gold_citations):
        if gold:
            scores.append(len(pred & gold) / len(gold))
        else:
            scores.append(1.0 if not pred else 0.0)
    return round(sum(scores) / len(scores), 4)


def citation_precision(pred_citations: list[set[str]],
                       gold_citations: list[set[str]]) -> float:
    """Average citation precision: what fraction of predicted citations are correct."""
    if not pred_citations:
        return 0.0
    scores = []
    for pred, gold in zip(pred_citations, gold_citations):
        if pred:
            scores.append(len(pred & gold) / len(pred))
        else:
            scores.append(1.0 if not gold else 0.0)
    return round(sum(scores) / len(scores), 4)
