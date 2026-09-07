"""Metric implementations for the M6 harness.

Moved here from `src/smartlawai/eval/metrics.py` (M6_ONBOARDING.md §4) and
extended to cover RESEARCH.md §5's full metric table.

Two rules govern every function below.

**Never hand-rolled where a real library exists** (RESEARCH.md §5). The
hand-rolled LCS ROUGE-L that used to live here is deleted; `rouge_scores()`
calls the `rouge-score` package. `bertscore_f1()` calls `bert-score`.

**Never fail open** (CLAUDE.md I2). A metric whose input is unavailable returns
`None`, not a substituted number. `0.0` is a measured result meaning "nothing
was correct"; `None` means "this was not measured". Conflating them is how the
old code fabricated metrics. Callers render `None` as `unavailable` with the
reason -- see `eval/report.py`.

Aggregate helpers return `MetricResult` so the reason travels with the number.

Imports nothing from `smartlawai`: this module is pure functions over data
structures (OPS.md §8, enforced by tests/test_eval_integration.py).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from eval import config


# --------------------------------------------------------------------------- #
# I2-shaped return type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricResult:
    """A metric value that knows whether it was actually measured.

    status:
      "ok"             -- `value` is a real measurement over `n` items.
      "unavailable"    -- could not be computed; `value` is None, `reason` says why.
      "not_applicable" -- the metric does not apply to this gold set / condition.
    """

    name: str
    value: float | None
    status: str = "ok"
    n: int = 0
    reason: str = ""
    detail: dict = field(default_factory=dict)

    @classmethod
    def unavailable(cls, name: str, reason: str, n: int = 0) -> MetricResult:
        return cls(name=name, value=None, status="unavailable", n=n, reason=reason)

    @classmethod
    def not_applicable(cls, name: str, reason: str) -> MetricResult:
        return cls(name=name, value=None, status="not_applicable", reason=reason)

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "status": self.status,
                "n": self.n, "reason": self.reason, "detail": self.detail}


# --------------------------------------------------------------------------- #
# Summarisation / answer text quality
# --------------------------------------------------------------------------- #
def rouge_scores(pred: str, ref: str) -> dict[str, float]:
    """ROUGE-1/2/L F1 via the `rouge-score` package.

    Replaces the hand-rolled LCS implementation this module used to carry
    (PLAN.md M6: "Delete the hand-rolled ROUGE-L."; RESEARCH.md §5: ROUGE is
    "never hand-rolled").

    Raises ImportError with an actionable message if the package is absent --
    it does NOT silently degrade to an approximation, which would be exactly
    the fabricated-metric failure I2 exists to prevent.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "rouge-score is required for rouge_scores(). Install the eval extras: "
            'pip install -e ".[eval]"'
        ) from e

    scorer = rouge_scorer.RougeScorer(list(config.ROUGE_VARIANTS), use_stemmer=True)
    scores = scorer.score(ref, pred)
    return {k: round(v.fmeasure, 4) for k, v in scores.items()}


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
    """BERTScore F1 (requires the bert-score package).

    Left exactly as inherited (M6_ONBOARDING.md §8: "already calls the real
    bert_score package correctly -- don't touch"). Note that bert-score pulls
    torch and is NOT installed in this repo today; no test has ever exercised
    this function. tests/test_metrics.py skips it with that reason stated
    rather than pretending it passes.
    """
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
# Retrieval metrics  (RESEARCH.md §5: "Recall@{5,10,20}, MRR, nDCG@10")
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


def _dcg(gains: list[float]) -> float:
    """Discounted cumulative gain with the standard log2(rank+1) discount."""
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(rankings: list[list[str]], relevances: list[dict[str, float]],
              k: int = config.NDCG_K) -> float:
    """nDCG@K over graded relevance judgments.

    relevances: per query, {chunk_id: gain}. Binary judgments are the special
    case where every gain is 1.0 -- pass {"c1": 1.0, ...} and this reduces to
    binary nDCG. Chunk ids absent from the dict score gain 0.

    A query whose ideal DCG is 0 (nothing judged relevant) contributes nothing
    and is excluded from the mean rather than counted as a 0.0 the system did
    not earn; if that leaves no queries at all, returns 0.0 and the caller's
    MetricResult wrapper reports n=0.
    """
    if not rankings:
        return 0.0
    scores: list[float] = []
    for ranked, rel in zip(rankings, relevances):
        ideal_gains = sorted(rel.values(), reverse=True)[:k]
        idcg = _dcg(ideal_gains)
        if idcg == 0:
            continue
        actual_gains = [rel.get(cid, 0.0) for cid in ranked[:k]]
        scores.append(_dcg(actual_gains) / idcg)
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)


def retrieval_metrics(rankings: list[list[str]],
                      relevances: list[dict[str, float]],
                      k_values: tuple[int, ...] = config.RETRIEVAL_K_VALUES,
                      ndcg_k: int = config.NDCG_K) -> list[MetricResult]:
    """The full RESEARCH.md §5 retrieval row for one system on one gold set.

    Returns `unavailable` results (not zeros) when there are no queries or no
    query carries a relevance judgment -- "we could not measure retrieval" and
    "retrieval scored zero" are different claims and the paper must not blur them.
    """
    if not rankings:
        return [MetricResult.unavailable(name, "no_ranked_results")
                for name in [f"recall@{k}" for k in k_values]
                + [f"precision@{k}" for k in k_values] + ["mrr", f"ndcg@{ndcg_k}"]]

    judged = [r for r in relevances if r]
    n = len(rankings)
    if not judged:
        return [MetricResult.unavailable(name, "no_relevance_judgments", n=n)
                for name in [f"recall@{k}" for k in k_values]
                + [f"precision@{k}" for k in k_values] + ["mrr", f"ndcg@{ndcg_k}"]]

    binary = [set(r.keys()) for r in relevances]
    out: list[MetricResult] = []
    for k in k_values:
        out.append(MetricResult(f"recall@{k}", recall_at_k(rankings, binary, k), n=n))
    for k in k_values:
        out.append(MetricResult(f"precision@{k}", precision_at_k(rankings, binary, k), n=n))
    out.append(MetricResult("mrr", mean_reciprocal_rank(rankings, binary), n=n))
    out.append(MetricResult(f"ndcg@{ndcg_k}", ndcg_at_k(rankings, relevances, ndcg_k), n=n))
    return out


# --------------------------------------------------------------------------- #
# Groundedness  (CLAUDE.md §5 -- ONE definition, used in serving, eval and paper)
# --------------------------------------------------------------------------- #
def groundedness_rate(labels: list[bool | None]) -> MetricResult:
    """Fraction of claims entailed by their cited passage.

    `labels[i] is None` means the verifier was unavailable for that claim
    (I2). Those claims are excluded from the denominator and counted in
    `detail["n_unavailable"]`, never scored as unsupported -- an unavailable
    verifier is not evidence of a hallucination.
    """
    judged = [x for x in labels if x is not None]
    n_unavailable = len(labels) - len(judged)
    if not judged:
        return MetricResult.unavailable("groundedness_rate",
                                        "no_judged_claims", n=len(labels))
    return MetricResult(
        "groundedness_rate", round(sum(judged) / len(judged), 4), n=len(judged),
        detail={"n_unavailable": n_unavailable, "n_total": len(labels)})


def hallucination_rate(labels: list[bool | None]) -> MetricResult:
    """% of generated claims NOT entailed by their cited passage.

    CLAUDE.md §5 permits exactly one definition of this number project-wide, so
    this is defined as the exact complement of `groundedness_rate` rather than
    recomputed independently. tests/test_metrics.py asserts the identity.
    """
    g = groundedness_rate(labels)
    if g.value is None:
        return MetricResult.unavailable("hallucination_rate", g.reason, n=g.n)
    return MetricResult("hallucination_rate", round(1.0 - g.value, 4),
                        n=g.n, detail=dict(g.detail))


# --------------------------------------------------------------------------- #
# Citation metrics
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


def citation_coverage(n_claims_with_citation: int, n_claims: int) -> MetricResult:
    """Fraction of generated claims that carry at least one citation."""
    if n_claims <= 0:
        return MetricResult.unavailable("citation_coverage", "no_claims_generated")
    return MetricResult("citation_coverage",
                        round(n_claims_with_citation / n_claims, 4), n=n_claims)


# ---- registry-backed metrics (RESEARCH.md §5; M4 supplies the resolution) ---- #
#
# These take ALREADY-RESOLVED registry results, deliberately. M6 must not import
# verify/registry_check.py -- that module belongs to M4 and does not exist yet,
# and creating the import edge would make M6 undeliverable until M4 lands
# (M6_ONBOARDING.md §8). The caller -- eventually M9's run_eval invocation -- is
# responsible for producing the resolved input once the registry exists.
#
# Expected shape per citation is M4's OWN published contract
# (M4_ONBOARDING.md §6's `verify/registry_check.py::RegistryResult`), reused
# verbatim rather than invented, so a real M4 result drops in with no
# translation layer the day that module lands:
#   {"status": "in_force" | "repealed" | "superseded" | "not_found",
#    "as_of": "<ISO date>",
#    "note": str | None}   # e.g. "repealed 2024-07-01; see BNS §318"
_VALID_REGISTRY_STATUSES = frozenset({"in_force", "repealed", "superseded", "not_found"})
_RESOLVED_STATUSES = frozenset({"in_force", "repealed", "superseded"})  # i.e. != not_found


def registry_coverage_rate(resolved: list[dict]) -> MetricResult:
    """Fraction of cited authorities the registry actually recognises at all.

    RESEARCH.md T6, verbatim: "Report coverage rate; out-of-registry citations
    flagged, not silently passed." `status == "not_found"` is M4's own honest
    "I don't know this citation" answer (M4_ONBOARDING §6: "'not_found' is a
    valid, honest return... never a guess dressed up as an answer") -- this
    metric is exactly the number that makes that limitation visible in every
    report rather than only in M4's own build-time notes.
    """
    if not resolved:
        return MetricResult.unavailable("registry_coverage_rate",
                                        "no_resolved_citations_supplied")
    covered = sum(1 for r in resolved if r.get("status") in _RESOLVED_STATUSES)
    return MetricResult("registry_coverage_rate", round(covered / len(resolved), 4),
                        n=len(resolved),
                        detail={"n_not_found": len(resolved) - covered})


def citation_validity_rate(resolved: list[dict]) -> MetricResult:
    """Fraction of cited authorities that resolve to a real statutory provision.

    "Valid" means the registry recognises the citation as pointing at something
    real -- `in_force`, `repealed`, or `superseded` all count, because all three
    mean the target exists; only `not_found` means it doesn't (or the registry
    simply doesn't cover it -- see `registry_coverage_rate` for that distinction
    reported separately, per RESEARCH.md T6, rather than conflated here).
    """
    if not resolved:
        return MetricResult.unavailable("citation_validity_rate",
                                        "no_resolved_citations_supplied")
    statuses = [r.get("status") for r in resolved]
    if any(s not in _VALID_REGISTRY_STATUSES for s in statuses):
        bad = [s for s in statuses if s not in _VALID_REGISTRY_STATUSES]
        return MetricResult.unavailable(
            "citation_validity_rate", f"malformed_registry_status:{bad[0]!r}",
            n=len(resolved))
    valid = sum(1 for s in statuses if s in _RESOLVED_STATUSES)
    return MetricResult("citation_validity_rate", round(valid / len(resolved), 4),
                        n=len(resolved),
                        detail={"n_not_found": len(resolved) - valid})


def repealed_citation_rate(resolved: list[dict]) -> MetricResult:
    """Fraction of EXISTING cited authorities that are no longer in force.

    The centrepiece number of RESEARCH.md §2's IPC->BNS natural experiment.
    Denominator is citations that resolve to something real (`in_force`,
    `repealed`, or `superseded`); `not_found` citations are excluded rather
    than counted as "in force" -- counting them as in-force would understate
    the repealed rate, the exact direction of error that would flatter this
    project's own result.
    """
    if not resolved:
        return MetricResult.unavailable("repealed_citation_rate",
                                        "no_resolved_citations_supplied")
    known = [r for r in resolved if r.get("status") in _RESOLVED_STATUSES]
    if not known:
        return MetricResult.unavailable("repealed_citation_rate",
                                        "registry_resolved_nothing", n=len(resolved))
    repealed = [r for r in known if r["status"] in ("repealed", "superseded")]
    return MetricResult(
        "repealed_citation_rate", round(len(repealed) / len(known), 4), n=len(known),
        detail={"n_not_found": len(resolved) - len(known),
                "n_total": len(resolved),
                "n_with_note": sum(1 for r in repealed if r.get("note"))})


def citation_support_rate(supported: list[bool | None]) -> MetricResult:
    """Fraction of citations whose cited passage actually supports the claim.

    Distinct from validity: a citation can name a real, in-force provision and
    still not support the sentence attached to it. `None` = not judged (I2).
    """
    judged = [x for x in supported if x is not None]
    if not judged:
        return MetricResult.unavailable("citation_support_rate",
                                        "no_judged_citations", n=len(supported))
    return MetricResult("citation_support_rate",
                        round(sum(judged) / len(judged), 4), n=len(judged),
                        detail={"n_unjudged": len(supported) - len(judged)})


# --------------------------------------------------------------------------- #
# Refusal / out-of-scope  (RESEARCH.md §5: "Precision / recall on labelled OOS")
# --------------------------------------------------------------------------- #
#
# Positive class = "the system should refuse". Per the gold OOS three-way label,
# both `out_of_scope` and `advice_seeking` are should-refuse; `in_scope` is not.
SHOULD_REFUSE_LABELS: frozenset[str] = frozenset({"out_of_scope", "advice_seeking"})


def refusal_metrics(gold_labels: list[str],
                    decisions: list[str]) -> list[MetricResult]:
    """Refusal precision / recall / F1 plus over-refusal on in-scope questions.

    gold_labels: "in_scope" | "out_of_scope" | "advice_seeking".
    decisions:   the pipeline's "ANSWER" | "REFUSE" outcome for the same items.
    """
    names = ["refusal_precision", "refusal_recall", "refusal_f1",
             "over_refusal_rate", "refusal_accuracy"]
    if not gold_labels or len(gold_labels) != len(decisions):
        return [MetricResult.unavailable(n, "no_or_misaligned_oos_items",
                                         n=len(gold_labels)) for n in names]

    tp = fp = fn = correct = 0
    n_in_scope = over_refused = 0
    for gold, decision in zip(gold_labels, decisions):
        should_refuse = gold in SHOULD_REFUSE_LABELS
        did_refuse = decision == "REFUSE"
        if should_refuse and did_refuse:
            tp += 1
        elif not should_refuse and did_refuse:
            fp += 1
        elif should_refuse and not did_refuse:
            fn += 1
        correct += int(should_refuse == did_refuse)
        if not should_refuse:
            n_in_scope += 1
            over_refused += int(did_refuse)

    p, r, f = prf(tp, fp, fn)
    n = len(gold_labels)
    results = [
        MetricResult("refusal_precision", p, n=n, detail={"tp": tp, "fp": fp, "fn": fn}),
        MetricResult("refusal_recall", r, n=n),
        MetricResult("refusal_f1", f, n=n),
    ]
    if n_in_scope:
        results.append(MetricResult("over_refusal_rate",
                                    round(over_refused / n_in_scope, 4), n=n_in_scope))
    else:
        results.append(MetricResult.unavailable("over_refusal_rate",
                                                "no_in_scope_items", n=n))
    results.append(MetricResult("refusal_accuracy", round(correct / n, 4), n=n))
    return results


# --------------------------------------------------------------------------- #
# Judge diagnostics  (RESEARCH.md §5: "Detection rate per perturbation type")
# --------------------------------------------------------------------------- #
def per_perturbation_detection_rate(
        perturbation_types: list[str],
        detected: list[bool | None]) -> dict[str, MetricResult]:
    """Per-type detection rate for the synthetic perturbation set.

    This is the diagnostic RESEARCH.md §2.5 cares about: an aggregate judge
    accuracy hides that a judge may catch fabricated holdings while missing
    every IPC->BNS substitution. `None` = judge unavailable for that item, and
    is excluded rather than counted as a miss (I2).
    """
    out: dict[str, MetricResult] = {}
    by_type: dict[str, list[bool | None]] = {}
    for ptype, det in zip(perturbation_types, detected):
        by_type.setdefault(ptype, []).append(det)
    for ptype, dets in sorted(by_type.items()):
        judged = [d for d in dets if d is not None]
        name = f"detection_rate[{ptype}]"
        if not judged:
            out[ptype] = MetricResult.unavailable(name, "judge_unavailable",
                                                  n=len(dets))
        else:
            out[ptype] = MetricResult(
                name, round(sum(judged) / len(judged), 4), n=len(judged),
                detail={"n_unjudged": len(dets) - len(judged)})
    return out


# --------------------------------------------------------------------------- #
# Cost / latency  (RESEARCH.md §5: "tokens, USD, p50/p95 per stage" via trace)
# --------------------------------------------------------------------------- #
def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile. `q` in [0, 1]. None for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return round(ordered[lo], 4)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 4)


def latency_summary(stage_durations_ms: dict[str, list[float]]) -> list[MetricResult]:
    """p50 / p95 wall-clock per pipeline stage, straight from PipelineTrace."""
    out: list[MetricResult] = []
    for stage, values in sorted(stage_durations_ms.items()):
        if not values:
            out.append(MetricResult.unavailable(f"latency_p50_ms[{stage}]",
                                                "stage_never_ran"))
            continue
        out.append(MetricResult(f"latency_p50_ms[{stage}]",
                                percentile(values, 0.50), n=len(values)))
        out.append(MetricResult(f"latency_p95_ms[{stage}]",
                                percentile(values, 0.95), n=len(values)))
    return out
