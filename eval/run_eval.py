"""The M6 orchestrator: gold sets in, a results table out.

    python -m eval.run_eval --split dev
    python -m eval.run_eval --split dev --baseline eval/results/dev_latest.json

M6_ONBOARDING.md §11:

    Runs Pipeline.ask() (unmodified) over every item in the named gold set(s)
    relevant to that split, computes every applicable metric from
    eval/metrics.py + eval/stats.py + eval/extractiveness.py, writes a results
    table to eval/results/{split}_{timestamp}.json. Never hand-types a number.

This is the ONE module in eval/ permitted to import `smartlawai` (OPS.md §8 says
the harness is pure functions over data structures; the onboarding requires the
runner to drive the real pipeline -- the boundary is drawn here, and
tests/test_eval_integration.py enforces that no other eval module crosses it).

Two things it does NOT do, deliberately:

  * It does not modify or reimplement any part of the retrieve/rerank/generate/
    verify/decide sequence. It calls the public `.ask(question, scope)` and
    reads the trace.
  * It does not invent a scope. Every ask carries an explicit `Scope` (I1). The
    eval corpus is ingested into a throwaway store so a measurement run can
    never read, or pollute, a real user's documents.

With every model still stubbed this produces a table of trivial or `unavailable`
scores. That is the expected floor: the done-when criterion is that the table
computes end to end and is honest about what it could not measure.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval import config, report
from eval.extractiveness import mean_extractiveness
from eval.gold_sets import describe_gold_set, load_all_gold_sets, sizing_note
from eval.metrics import (
    MetricResult,
    groundedness_rate,
    hallucination_rate,
    latency_summary,
    refusal_metrics,
    retrieval_metrics,
    rouge_scores,
)
from eval.stats import expected_calibration_error, reliability_diagram_data

HARNESS_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Per-item record
# --------------------------------------------------------------------------- #
@dataclass
class ItemResult:
    """Everything one gold item produced. Written to the results JSON verbatim
    so any aggregate can be recomputed without re-running the pipeline."""

    item_id: str
    task: str
    question: str = ""
    decision: str = ""
    answer: str = ""
    top_chunk_ids: list[str] = field(default_factory=list)
    n_candidates: int = 0
    n_claims: int = 0
    verification: list[dict] = field(default_factory=list)
    stage_durations_ms: dict[str, float] = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    trace_id: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "task": self.task,
                "question": self.question, "decision": self.decision,
                "answer": self.answer, "top_chunk_ids": self.top_chunk_ids,
                "n_candidates": self.n_candidates, "n_claims": self.n_claims,
                "verification": self.verification,
                "stage_durations_ms": self.stage_durations_ms,
                "cost": self.cost, "trace_id": self.trace_id, "error": self.error}


# --------------------------------------------------------------------------- #
# Corpus + pipeline setup
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=_REPO_ROOT, capture_output=True, text=True,
                             timeout=10, check=False)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- git absent/unavailable is not fatal, just "unknown"
        return "unknown"


def _split_version() -> tuple[str | None, str]:
    """OPS.md §4: "Every RunManifest and every results file records
    split_version." `data/splits/*.json` (I5's frozen document-level splits)
    do not exist yet (M1 not landed), so there is nothing to read. Returns
    (version_or_None, note) -- the field is always present in the manifest,
    explicitly null with a reason rather than silently omitted, matching this
    project's established convention for an honest absence.
    """
    train_split = config.SPLITS_DIR / "train.json"
    if not train_split.exists():
        return None, "data/splits/train.json does not exist yet (M1 not landed)"
    try:
        data = json.loads(train_split.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return None, f"data/splits/train.json unreadable: {e}"
    version = data.get("split_version") or data.get("version")
    if version is None:
        return None, "data/splits/train.json exists but carries no version field"
    return str(version), ""


class EvalCorpus:
    """A throwaway document store for one evaluation run.

    Isolation is not a convenience here. Pointing the harness at a real store
    would let a measurement run write chunks into it, and would let gold-item
    scopes address documents the run did not ingest -- both of which silently
    corrupt the thing being measured.
    """

    def __init__(self, corpus_paths: list[Path], owner_id: str = config.EVAL_OWNER_ID):
        self.owner_id = owner_id
        self.corpus_paths = corpus_paths
        self._tmp = tempfile.TemporaryDirectory(prefix="smartlaw-eval-")
        self.doc_ids: list[str] = []
        self._previous_dir = os.environ.get("SMARTLAW_LOCAL_DIR")
        os.environ["SMARTLAW_LOCAL_DIR"] = self._tmp.name

        from smartlawai.adapters.factory import get_backend
        from smartlawai.pipeline import Pipeline

        self.backend = get_backend("local")
        self.pipeline = Pipeline(self.backend)
        for path in corpus_paths:
            doc_id, _ = self.pipeline.ingest(str(path), doc_type="JUDGMENT",
                                             source="eval", owner_id=owner_id)
            self.doc_ids.append(doc_id)

    def scope_for(self, doc_ids: list[str] | None = None):
        """I1: an explicit Scope, always. Gold items that name their own
        doc_ids get them; items that do not are scoped to the whole eval
        corpus, which is still an explicit, enumerated list -- never a
        'search everything' path."""
        from smartlawai.scope import Scope
        ids = [d for d in (doc_ids or []) if d in self.doc_ids] or self.doc_ids
        return Scope(doc_ids=tuple(ids), owner_id=self.owner_id)

    def close(self) -> None:
        if self._previous_dir is None:
            os.environ.pop("SMARTLAW_LOCAL_DIR", None)
        else:
            os.environ["SMARTLAW_LOCAL_DIR"] = self._previous_dir
        self._tmp.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def default_corpus() -> list[Path]:
    """Structural smoke corpus.

    M4_ONBOARDING.md §4's convention, adopted here: sample_docs/ is for
    structural testing only, never for a reported number. The results manifest
    records which corpus was used so a reader can tell the difference.
    """
    sample = _REPO_ROOT / "sample_docs" / "sample_judgment.txt"
    return [sample] if sample.exists() else []


# --------------------------------------------------------------------------- #
# Running one gold set
# --------------------------------------------------------------------------- #
def _ask(corpus: EvalCorpus, item_id: str, task: str, question: str,
         doc_ids: list[str]) -> ItemResult:
    result = ItemResult(item_id=item_id, task=task, question=question)
    try:
        answer = corpus.pipeline.ask(question, corpus.scope_for(doc_ids))
    except Exception as e:  # noqa: BLE001 -- a failing item is data, not a reason to abort the run
        result.error = f"{type(e).__name__}: {e}"
        return result

    trace = answer.trace
    result.decision = answer.decision
    result.answer = answer.answer
    result.trace_id = trace.trace_id
    result.top_chunk_ids = list(trace.retrieval.get("top_chunk_ids", []))
    result.n_candidates = int(trace.retrieval.get("n_candidates", 0))
    result.n_claims = int(trace.generation.get("n_claims", 0))
    result.verification = list(trace.verification.get("results", []))
    result.stage_durations_ms = {s.name: s.duration_ms for s in trace.stages}
    result.cost = dict(trace.cost)
    for stage in trace.stages:
        if stage.status == "error":
            result.error = f"stage {stage.name}: {stage.error}"
    return result


def run_retrieval(corpus: EvalCorpus, items: list) -> tuple[list[ItemResult],
                                                            list[MetricResult]]:
    results = [_ask(corpus, i.item_id, "retrieval", i.query, list(i.doc_ids))
               for i in items]
    rankings = [r.top_chunk_ids for r in results]
    relevances = [i.gains() for i in items]
    return results, retrieval_metrics(rankings, relevances)


def run_oos(corpus: EvalCorpus, items: list) -> tuple[list[ItemResult],
                                                      list[MetricResult]]:
    results = [_ask(corpus, i.item_id, "oos", i.question, list(i.doc_ids))
               for i in items]
    gold = [i.label for i, r in zip(items, results) if r.error is None]
    decisions = [r.decision for r in results if r.error is None]
    return results, refusal_metrics(gold, decisions)


def _prefixed(metrics: list[MetricResult], task: str) -> list[MetricResult]:
    """Namespace a set's metrics as `<task>.<metric>`.

    groundedness.jsonl and entailment.jsonl produce the same metric names for
    deliberately different populations (the evaluation set vs the held-out
    judge-evaluation set). Emitting both as bare `hallucination_rate` would put
    two different numbers under one label in the report and collapse them to one
    entry in the regression gate's flat view -- the two sets must never be
    conflated (RESEARCH.md §2.5), and the metric names are where that starts.
    """
    return [MetricResult(f"{task}.{m.name}", m.value, m.status, m.n, m.reason,
                         m.detail) for m in metrics]


def _verifier_predictions(corpus: EvalCorpus, items: list) -> list[bool | None]:
    """Run each gold (claim, passage) pair through the pipeline's OWN verifier
    -- a real, existing system component that stands in for the not-yet-built
    HHEM-2.1/InLegalNLI -- rather than reading the human label as if it were
    the system's own output.

    Corrected during audit (see docs/M6_GAP_ANALYSIS.md G1): the earlier
    version computed groundedness_rate directly from the gold label, which
    measures the gold set's own composition, not the system -- a fabricated-
    metric bug of the exact kind CLAUDE.md I2 exists to prevent.

    `None` when the verifier reports `status != "ok"` (I2: an unavailable
    verifier is not evidence either way).
    """
    from smartlawai.adapters.base import Chunk, RetrievedChunk
    from smartlawai.config import FAITHFULNESS_THRESHOLD
    from smartlawai.protocols import Claim

    predictions: list[bool | None] = []
    for item in items:
        chunk = Chunk(chunk_id=f"gold-{item.item_id}", doc_id=item.source_doc_id or "gold",
                      chunk_index=0, chunk_text=item.passage, char_start=0,
                      char_end=len(item.passage))
        claim = Claim(text=item.claim_text, passage_ids=[chunk.chunk_id])
        result = corpus.pipeline.verifier.verify(claim, [RetrievedChunk(chunk=chunk, score=1.0)])
        if result.status != "ok" or result.score is None:
            predictions.append(None)
        else:
            predictions.append(result.score >= FAITHFULNESS_THRESHOLD)
    return predictions


def score_groundedness(corpus: EvalCorpus, items: list,
                       task: str = "groundedness") -> list[MetricResult]:
    """Claim-level groundedness, judge-based (CLAUDE.md §5: "judge-based",
    RESEARCH.md §5: "InLegalNLI + HHEM-2.1, both reported").

    Two genuinely different things are computed and reported separately, per
    RESEARCH.md §5's own split between the "QA" row (system groundedness) and
    the "Judge" row (agreement with human labels) -- conflating them was the
    audit's G1 finding:

      1. `groundedness_rate`/`hallucination_rate` FROM THE VERIFIER'S OWN
         PREDICTIONS -- honestly trivial under the current stub verifier
         (which always reports "supported"), but a real measurement of the
         wired-in component, not an echo of the gold label.
      2. `judge_accuracy_vs_human` / `judge_agreement_kappa` -- does the
         verifier's prediction agree with the human gold label? This is what
         groundedness.jsonl/entailment.jsonl actually exist to produce
         (RESEARCH.md §5 "Judge | Agreement with human labels, Cohen's kappa").

    `citation_coverage` was removed from this function during the audit: gold
    rows carry no real system-generated `citations` (the field is always empty
    in the shipped seed data), so computing it here was a category error, not
    a metric -- it measured an unpopulated fixture field, not system output.
    """
    from eval.annotate.agreement import cohen_kappa

    predictions = _verifier_predictions(corpus, items)
    gold = [i.supported() for i in items]

    metrics = [groundedness_rate(predictions), hallucination_rate(predictions)]

    paired = [(p, g) for p, g in zip(predictions, gold) if p is not None and g is not None]
    if paired:
        kappa = cohen_kappa([p for p, _ in paired], [g for _, g in paired])
        accuracy = sum(p == g for p, g in paired) / len(paired)
        metrics.append(MetricResult("judge_agreement_kappa", kappa.value,
                                    status=kappa.status, n=kappa.n_items,
                                    reason=kappa.reason))
        metrics.append(MetricResult("judge_accuracy_vs_human", round(accuracy, 4),
                                    n=len(paired)))
    else:
        metrics.append(MetricResult.unavailable(
            "judge_agreement_kappa", "no_paired_verifier_prediction_and_gold_label"))
        metrics.append(MetricResult.unavailable(
            "judge_accuracy_vs_human", "no_paired_verifier_prediction_and_gold_label"))

    # RESEARCH.md §5: faithfulness is never reported without extractiveness.
    # This IS a legitimate pure-text computation -- claim_text and passage are
    # both real gold-row text, no system involvement required to measure it.
    ext = mean_extractiveness([i.claim_text for i in items],
                              [i.passage for i in items])
    metrics.append(
        MetricResult(f"extractiveness_{config.EXTRACTIVENESS_NGRAM}gram", ext,
                     n=len(items))
        if ext is not None else
        MetricResult.unavailable(f"extractiveness_{config.EXTRACTIVENESS_NGRAM}gram",
                                 "no_claim_long_enough_to_measure", n=len(items)))

    rouge_available = True
    try:
        rouge = [rouge_scores(i.claim_text, i.passage) for i in items]
    except ImportError:
        rouge_available = False
    if rouge_available and rouge:
        for variant in config.ROUGE_VARIANTS:
            metrics.append(MetricResult(
                f"{variant}_claim_vs_passage",
                round(sum(r[variant] for r in rouge) / len(rouge), 4), n=len(rouge)))
    elif not rouge_available:
        for variant in config.ROUGE_VARIANTS:
            metrics.append(MetricResult.unavailable(
                f"{variant}_claim_vs_passage",
                "rouge_score_package_not_installed", n=len(items)))

    return _prefixed(metrics, task)


def score_repealed(items: list) -> list[MetricResult]:
    """Authority metrics -- correctly `unavailable`, not fabricated.

    Corrected during audit (see docs/M6_GAP_ANALYSIS.md G2): the earlier
    version called `item.as_registry_result()` and fed the gold label back in
    as if it were a real registry answer, producing a "measured" number that
    was actually the gold set's own label distribution. M6_ONBOARDING.md §8 is
    explicit that these metrics "can't be computed against a real registry
    until M4 lands" -- so until it does, the honest report is `unavailable`,
    matching CLAUDE.md I2 exactly (a missing component reports its absence,
    never a substituted number).

    `RepealedItem.as_registry_result()` still exists as a metric-testing
    fixture (see tests/test_metrics.py) -- it is simply no longer called from
    a live run.
    """
    n = len(items)
    reason = "no_registry_available_M4_not_started"
    return [MetricResult.unavailable("registry_coverage_rate", reason, n=n),
            MetricResult.unavailable("citation_validity_rate", reason, n=n),
            MetricResult.unavailable("repealed_citation_rate", reason, n=n)]


def calibration_metrics(item_results: list[ItemResult],
                        correctness_by_item: dict[str, bool] | None = None,
                        ) -> tuple[list[MetricResult], list[dict]]:
    """ECE + reliability bins for the faithfulness score the product displays.

    Calibration asks whether a displayed 0.8 means 80% correct, so it needs two
    independent things: the verifier's confidence, and a GROUND-TRUTH correctness
    label for the same item. `correctness_by_item` supplies the second.

    When it is absent, this returns `unavailable` rather than a number. The
    obvious substitute -- scoring the gate's own ACCEPT/REFUSE outcome as
    "correct" -- is circular: the gate refuses precisely when the verifier's
    confidence is low, so confidence would predict itself and every system would
    look perfectly calibrated. That fabricated 0.0 is exactly the class of bug
    I2 exists to prevent, and RESEARCH.md §5 makes calibration mandatory because
    the number reaches non-expert users making real decisions. It has to be a
    real measurement or no measurement.

    I2 also governs the confidences themselves: a verification with
    `status != "ok"` has no score and is excluded, never read as 0.0 or 0.5.
    The excluded count is reported so a run where the verifier was mostly
    unavailable cannot be mistaken for a well-calibrated one.
    """
    confidences: list[float] = []
    correctness: list[bool] = []
    n_unavailable = 0
    n_unlabelled = 0

    for item in item_results:
        gold_correct = (correctness_by_item or {}).get(item.item_id)
        for v in item.verification:
            if v.get("status") != "ok" or v.get("score") is None:
                n_unavailable += 1
                continue
            if gold_correct is None:
                n_unlabelled += 1
                continue
            confidences.append(float(v["score"]))
            correctness.append(bool(gold_correct))

    if not confidences:
        reason = ("no_gold_correctness_labels_for_generated_claims"
                  if n_unlabelled else "no_verifier_scores_available")
        return ([MetricResult.unavailable(
            "ece", reason, n=n_unavailable + n_unlabelled)], [])

    bins = reliability_diagram_data(confidences, correctness)
    ece = expected_calibration_error(confidences, correctness)
    return ([MetricResult("ece", ece, n=len(confidences),
                          detail={"n_unavailable_verifications": n_unavailable,
                                  "n_without_gold_label": n_unlabelled})],
            bins)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(split: str = "dev", out_dir: Path | None = None,
        gold_dir: Path | None = None, corpus_paths: list[Path] | None = None,
        seed: int = config.DEFAULT_SEED, write: bool = True) -> dict:
    """Run the pipeline over every available gold set and write the results table."""
    out_dir = out_dir or config.RESULTS_DIR
    corpus_paths = corpus_paths if corpus_paths is not None else default_corpus()

    gold, missing = load_all_gold_sets(gold_dir)
    started = datetime.now(UTC)
    split_version, split_version_note = _split_version()

    sections: dict[str, list[dict]] = {"retrieval": [], "answer": [], "citation": [],
                                       "robustness": [], "calibration": [],
                                       "cost": [], "human": []}
    item_results: list[ItemResult] = []
    reliability: list[dict] = []
    gold_stats: dict[str, dict] = {}
    notes: list[str] = []

    with EvalCorpus(corpus_paths) as corpus:
        manifest_extra = {
            "backend": type(corpus.backend).__name__,
            "generator_model": type(corpus.pipeline.generator).__name__,
            "encoder": type(corpus.pipeline.encoder).__name__,
            "reranker": type(corpus.pipeline.reranker).__name__,
            "verifier": type(corpus.pipeline.verifier).__name__,
            "pipeline_is_stubbed": type(corpus.pipeline.generator).__name__.startswith(
                "Stub"),
            "corpus_documents": [p.name for p in corpus_paths],
            "n_corpus_documents": len(corpus.doc_ids),
        }

        for name, items in gold.items():
            stats = describe_gold_set(name, items, gold_dir)
            gold_stats[name] = stats.to_dict()
            notes.append(sizing_note(stats))
            if not items:
                continue

            if name == "retrieval":
                results, metrics = run_retrieval(corpus, items)
                item_results += results
                sections["retrieval"] += report.as_dicts(metrics)
            elif name == "oos":
                results, metrics = run_oos(corpus, items)
                item_results += results
                sections["robustness"] += report.as_dicts(metrics)
            elif name in ("groundedness", "entailment"):
                sections["answer"] += report.as_dicts(
                    score_groundedness(corpus, items, name))
            elif name == "repealed":
                sections["citation"] += report.as_dicts(score_repealed(items))

        cal_metrics, reliability = calibration_metrics(item_results)
        sections["calibration"] += report.as_dicts(cal_metrics)

    # ---- cost / latency, straight from PipelineTrace ----
    durations: dict[str, list[float]] = {}
    for item in item_results:
        for stage, ms in item.stage_durations_ms.items():
            durations.setdefault(stage, []).append(ms)
    sections["cost"] += report.as_dicts(latency_summary(durations))

    # ---- human evaluation summary (annotations, if any were collected) ----
    sections["human"] += report.as_dicts(_human_metrics(gold))

    failures = [{"item_id": i.item_id, "stage": i.task, "error": i.error}
                for i in item_results if i.error]

    results: dict[str, Any] = {
        "manifest": {
            "harness_version": HARNESS_VERSION,
            "split": split,
            "split_version": split_version,
            "split_version_note": split_version_note,
            "seed": seed,
            "generated_at": started.isoformat(),
            "duration_s": round((datetime.now(UTC) - started).total_seconds(), 3),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "n_items_run": len(item_results),
            "gold_set_checksums": {n: s["sha256"] for n, s in gold_stats.items()},
            **manifest_extra,
        },
        "gold_sets": gold_stats,
        "missing_gold_sets": missing,
        "sizing_notes": notes,
        "sections": sections,
        "reliability_diagram": reliability,
        "items": [i.to_dict() for i in item_results],
        "failures": failures,
        "error_analysis": _error_analysis(item_results),
        "summary": {
            "n_gold_sets_loaded": len(gold),
            "n_gold_sets_missing": len(missing),
            "n_items_run": len(item_results),
            "n_failures": len(failures),
            "n_metrics_computed": sum(
                1 for ms in sections.values() for m in ms if m["status"] == "ok"),
            "n_metrics_unavailable": sum(
                1 for ms in sections.values() for m in ms if m["status"] != "ok"),
        },
    }

    if write:
        stem = f"{split}_{started.strftime('%Y%m%dT%H%M%SZ')}"
        json_path, md_path = report.write_report(results, out_dir, stem)
        results["manifest"]["json_path"] = str(json_path)
        results["manifest"]["markdown_path"] = str(md_path)
    return results


def _human_metrics(gold: dict[str, list]) -> list[MetricResult]:
    """Agreement and duplication actually achieved per set (RESEARCH.md §7.5:
    report agreement for every study)."""
    from eval.schemas import GOLD_SET_SPECS

    out: list[MetricResult] = []
    for name, items in sorted(gold.items()):
        spec = GOLD_SET_SPECS[name]
        agreements = [i.agreement for i in items if getattr(i, "agreement", None)
                      is not None]
        if agreements:
            out.append(MetricResult(f"agreement[{name}]",
                                    round(sum(agreements) / len(agreements), 4),
                                    n=len(agreements)))
        elif spec.iaa_duplication_rate == 0.0:
            out.append(MetricResult.not_applicable(
                f"agreement[{name}]",
                "0% duplication by design -- objective registry lookup "
                "(RESEARCH.md §7.1.3)"))
        else:
            out.append(MetricResult.unavailable(
                f"agreement[{name}]", "no_item_carries_a_duplicate_annotation",
                n=len(items)))
    return out


def _error_analysis(item_results: list[ItemResult]) -> dict[str, dict[str, int]]:
    """Group outcomes by the reasons the pipeline itself recorded."""
    decisions: dict[str, int] = {}
    verifier_reasons: dict[str, int] = {}
    errors: dict[str, int] = {}
    for item in item_results:
        if item.decision:
            decisions[item.decision] = decisions.get(item.decision, 0) + 1
        for v in item.verification:
            key = f"{v.get('status', '?')}:{v.get('reason', '')}"
            verifier_reasons[key] = verifier_reasons.get(key, 0) + 1
        if item.error:
            key = item.error.split(":")[0]
            errors[key] = errors.get(key, 0) + 1
    out = {"decisions": decisions, "verification_reasons": verifier_reasons}
    if errors:
        out["errors"] = errors
    return out


# --------------------------------------------------------------------------- #
# CLI + regression gate
# --------------------------------------------------------------------------- #
def check_regression(results: dict, baseline_path: Path,
                     tolerance: float = config.REGRESSION_TOLERANCE) -> int:
    """CI eval-regression gate (M6_ONBOARDING.md §11).

    Deliberately generous while every model is a stub: this guards against the
    harness silently breaking, not against quality regressions that cannot
    exist yet. Tightening it as real components land is a later milestone's
    call. Returns a process exit code.
    """
    if not baseline_path.exists():
        print(f"[regression] no baseline at {baseline_path}; nothing to compare. "
              f"Commit this run's JSON as the baseline.")
        return 0
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    regressions, disappeared = report.compare_to_baseline(results, baseline, tolerance)

    for r in regressions:
        print(f"[regression] {r['metric']}: {r['baseline']} -> {r['current']} "
              f"({r['delta']:+.4f}, tolerance {tolerance})")
    for d in disappeared:
        print(f"[regression] {d['metric']} was measured in the baseline "
              f"({d['baseline']}) and is not measured now -- the harness may have "
              f"stopped computing it.")
    if regressions or disappeared:
        return 1
    print(f"[regression] OK: no metric regressed past {tolerance} vs {baseline_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev", choices=list(config.SPLITS))
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--gold-dir", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, nargs="*", default=None,
                        help="documents to ingest; defaults to sample_docs/")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--baseline", type=Path, default=None,
                        help="results JSON to compare against; exit 1 on regression")
    parser.add_argument("--tolerance", type=float,
                        default=config.REGRESSION_TOLERANCE)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    results = run(split=args.split, out_dir=args.out_dir, gold_dir=args.gold_dir,
                  corpus_paths=list(args.corpus) if args.corpus else None,
                  seed=args.seed, write=not args.no_write)

    summary = results["summary"]
    print(f"split={args.split} items={summary['n_items_run']} "
          f"metrics_ok={summary['n_metrics_computed']} "
          f"metrics_unavailable={summary['n_metrics_unavailable']} "
          f"failures={summary['n_failures']}")
    if not args.no_write:
        print(f"wrote {results['manifest']['json_path']}")
        print(f"wrote {results['manifest']['markdown_path']}")

    if args.baseline:
        return check_regression(results, args.baseline, args.tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
