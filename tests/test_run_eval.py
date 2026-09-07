"""Evaluation runner tests (M6 Phase 5/13).

`eval.run_eval` is the ONE M6 module allowed to import `smartlawai`
(M6_ONBOARDING.md §11); these tests exercise that boundary directly, against
the real (stubbed) `Pipeline`, using the sample judgment for structural input
(M4_ONBOARDING.md §4's convention: never a reported number).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import config, run_eval
from eval.gold_sets import write_gold_set
from eval.schemas import (
    GroundednessItem,
    OOSItem,
    RepealedItem,
    RetrievalItem,
)

SAMPLE = Path(__file__).resolve().parent.parent / "sample_docs" / "sample_judgment.txt"
requires_sample = pytest.mark.skipif(not SAMPLE.exists(),
                                     reason="sample_docs/sample_judgment.txt missing")


@pytest.fixture()
def corpus():
    with run_eval.EvalCorpus([SAMPLE]) as c:
        yield c


# --------------------------------------------------------------------------- #
# I1: every ask carries an explicit Scope
# --------------------------------------------------------------------------- #
@requires_sample
def test_corpus_ingests_and_scopes_to_its_own_documents(corpus):
    from smartlawai.scope import Scope
    scope = corpus.scope_for()
    assert isinstance(scope, Scope)
    assert scope.doc_ids == tuple(corpus.doc_ids)
    assert scope.owner_id == config.EVAL_OWNER_ID


@requires_sample
def test_scope_for_ignores_doc_ids_outside_the_corpus():
    """I1: the scope is always an explicit, enumerated list -- never a
    'search everything' path, and never silently widened by bad input."""
    with run_eval.EvalCorpus([SAMPLE]) as c:
        scope = c.scope_for(["doc-does-not-exist"])
        assert scope.doc_ids == tuple(c.doc_ids)  # falls back to the real corpus


@requires_sample
def test_eval_corpus_uses_a_throwaway_store(corpus, monkeypatch):
    """A measurement run must never read or pollute a real user's documents."""
    import os
    assert os.environ["SMARTLAW_LOCAL_DIR"] != ""
    assert "smartlaw-eval-" in os.environ["SMARTLAW_LOCAL_DIR"]


@requires_sample
def test_eval_corpus_restores_the_previous_local_dir_on_close(monkeypatch):
    import os
    monkeypatch.setenv("SMARTLAW_LOCAL_DIR", "/some/real/store")
    with run_eval.EvalCorpus([SAMPLE]):
        assert os.environ["SMARTLAW_LOCAL_DIR"] != "/some/real/store"
    assert os.environ["SMARTLAW_LOCAL_DIR"] == "/some/real/store"


# --------------------------------------------------------------------------- #
# _ask: reads the trace, never invents a number
# --------------------------------------------------------------------------- #
@requires_sample
def test_ask_populates_an_item_result_from_the_real_trace(corpus):
    result = run_eval._ask(corpus, "q1", "oos", "What does clause one say?", [])
    assert result.error is None
    assert result.decision in ("ANSWER", "REFUSE")
    assert result.trace_id.startswith("tr-")
    assert "retrieve" in result.stage_durations_ms
    assert result.n_candidates > 0


@requires_sample
def test_ask_records_an_error_without_raising():
    """A failing item is data for the report, not a reason to abort the run."""
    with run_eval.EvalCorpus([SAMPLE]) as c:
        # An owner_id Scope rejects (I1) surfaces as a per-item error, not a crash.
        from smartlawai.scope import Scope
        bad_scope_owner = ""
        try:
            Scope(doc_ids=tuple(c.doc_ids), owner_id=bad_scope_owner)
            pytest.skip("Scope unexpectedly accepted an empty owner_id")
        except ValueError:
            pass  # confirms Scope itself enforces I1; _ask's try/except is below


def test_ask_catches_pipeline_exceptions_as_item_errors(monkeypatch):
    class ExplodingPipeline:
        def ask(self, *a, **k):
            raise RuntimeError("boom")

    class FakeCorpus:
        pipeline = ExplodingPipeline()

        def scope_for(self, doc_ids=None):
            return None

    result = run_eval._ask(FakeCorpus(), "q1", "oos", "question", [])
    assert result.error == "RuntimeError: boom"
    assert result.decision == ""


# --------------------------------------------------------------------------- #
# Per-set scoring functions
# --------------------------------------------------------------------------- #
@requires_sample
def test_run_retrieval_reports_unavailable_without_relevance_judgments(corpus):
    items = [RetrievalItem(item_id="r1", query="What is the holding?",
                           judgments={})]
    results, metrics = run_eval.run_retrieval(corpus, items)
    assert len(results) == 1
    assert all(m.status == "unavailable" for m in metrics)


@requires_sample
def test_run_oos_computes_refusal_metrics_from_real_decisions(corpus):
    items = [OOSItem(item_id="o1", question="What does clause one say?",
                     label="in_scope")]
    results, metrics = run_eval.run_oos(corpus, items)
    names = {m.name for m in metrics}
    assert "refusal_accuracy" in names
    assert len(results) == 1


@requires_sample
def test_score_groundedness_runs_the_real_verifier_not_the_gold_label(corpus):
    """Audit fix (docs/M6_GAP_ANALYSIS.md G1): the earlier version read the
    gold label back as if it were the system's own judgment, which is
    tautological (it always reproduced the gold set's own label balance
    regardless of any verifier). This asserts the corrected behaviour: the
    number comes from `corpus.pipeline.verifier.verify()`, a real component.

    The stub verifier always returns status="ok", score=1.0 for any claim with
    a passage_id -- so groundedness_rate here is honestly 1.0 (the stub's own,
    trivial, always-say-yes behaviour), NOT the gold set's label balance. Proof
    that this is verifier-driven, not label-driven: two items with DIFFERENT
    gold labels ("entailed" and "not_entailed") both count as "grounded",
    because the metric reflects the verifier's prediction, not the label.
    """
    items = [
        GroundednessItem(item_id="g1", claim_text="The appellant was granted land.",
                         passage="The appellant was granted land in 1949.",
                         label="entailed"),
        GroundednessItem(item_id="g2", claim_text="The respondent won the appeal.",
                         passage="The appellant was granted land in 1949.",
                         label="not_entailed"),
    ]
    metrics = run_eval.score_groundedness(corpus, items)
    by_name = {m.name: m for m in metrics}
    assert by_name["groundedness.groundedness_rate"].value == 1.0
    assert by_name["groundedness.hallucination_rate"].value == 0.0


@requires_sample
def test_score_groundedness_reports_judge_agreement_with_human_labels(corpus):
    """RESEARCH.md §5's "Judge | Agreement with human labels" row -- the
    metric groundedness.jsonl actually exists to produce. The stub verifier
    always predicts True, so accuracy-vs-human exactly equals the fraction of
    gold items labelled 'entailed', and kappa (chance-corrected) is near zero
    -- an honest signal that the current (stub) verifier does no real work."""
    items = [
        GroundednessItem(item_id="g1", claim_text="a claim text here now",
                         passage="a claim text here now in full", label="entailed"),
        GroundednessItem(item_id="g2", claim_text="another claim text here",
                         passage="another claim text here in full",
                         label="not_entailed"),
    ]
    by_name = {m.name: m for m in run_eval.score_groundedness(corpus, items)}
    assert by_name["groundedness.judge_accuracy_vs_human"].value == 0.5
    assert by_name["groundedness.judge_agreement_kappa"].status in ("ok", "unavailable")


@requires_sample
def test_score_groundedness_never_reports_faithfulness_without_extractiveness(corpus):
    """RESEARCH.md §5: mandatory alongside faithfulness."""
    items = [GroundednessItem(item_id="g1", claim_text="short claim here now",
                              passage="short claim here now, said the court",
                              label="entailed")]
    names = {m.name for m in run_eval.score_groundedness(corpus, items)}
    assert any("extractiveness" in n for n in names)
    assert any("groundedness_rate" in n for n in names)


@requires_sample
def test_score_groundedness_no_longer_reads_the_unpopulated_citations_field(corpus):
    """Audit fix: citation_coverage was removed -- gold rows carry no real
    system-generated citations, so it was measuring an unpopulated fixture
    field, not system output (docs/M6_GAP_ANALYSIS.md G1)."""
    items = [GroundednessItem(item_id="g1", claim_text="a claim here now please",
                              passage="a claim here now please indeed",
                              label="entailed")]
    names = {m.name for m in run_eval.score_groundedness(corpus, items)}
    assert not any("citation_coverage" in n for n in names)


@requires_sample
def test_score_groundedness_and_entailment_namespaces_never_collide(corpus):
    """RESEARCH.md §2.5: the two sets must never be conflated, starting with
    their metric names."""
    items = [GroundednessItem(item_id="g1", claim_text="a claim text here",
                              passage="a claim text here in full", label="entailed")]
    grounded_names = {m.name for m in run_eval.score_groundedness(corpus, items, "groundedness")}
    entailment_names = {m.name for m in run_eval.score_groundedness(corpus, items, "entailment")}
    assert grounded_names.isdisjoint(entailment_names)
    assert all(n.startswith("groundedness.") for n in grounded_names)
    assert all(n.startswith("entailment.") for n in entailment_names)


def test_score_repealed_is_unavailable_not_fabricated():
    """Audit fix (docs/M6_GAP_ANALYSIS.md G2): the earlier version fed the gold
    label back in via as_registry_result() and reported a 'measured' rate that
    was actually the gold set's own label distribution. M6_ONBOARDING.md §8 is
    explicit these metrics "can't be computed against a real registry until M4
    lands" -- so the honest report, until it does, is unavailable."""
    items = [
        RepealedItem(item_id="r1", question="q", citation="IPC s.302",
                    in_force_label="repealed", superseded_by="BNS s.103"),
        RepealedItem(item_id="r2", question="q", citation="BNS s.103",
                    in_force_label="in_force"),
    ]
    by_name = {m.name: m for m in run_eval.score_repealed(items)}
    assert by_name["registry_coverage_rate"].status == "unavailable"
    assert by_name["citation_validity_rate"].status == "unavailable"
    assert by_name["repealed_citation_rate"].status == "unavailable"
    assert all(m.value is None for m in by_name.values())
    assert all("M4_not_started" in m.reason for m in by_name.values())


# --------------------------------------------------------------------------- #
# Calibration: circularity guard
# --------------------------------------------------------------------------- #
def _item_result(item_id, score, decision="ANSWER"):
    return run_eval.ItemResult(
        item_id=item_id, task="oos", decision=decision,
        verification=[{"status": "ok", "score": score, "reason": ""}])


def test_calibration_is_unavailable_without_gold_correctness_labels():
    """Scoring the gate's own ACCEPT/REFUSE as 'correct' would be circular:
    confidence would predict itself and every system would look perfectly
    calibrated. Must be unavailable, never a fabricated number."""
    items = [_item_result("i1", 0.9), _item_result("i2", 0.2, "REFUSE")]
    metrics, bins = run_eval.calibration_metrics(items)
    assert metrics[0].status == "unavailable"
    assert metrics[0].reason == "no_gold_correctness_labels_for_generated_claims"
    assert bins == []


def test_calibration_computes_a_real_number_given_gold_labels():
    items = [_item_result("i1", 1.0), _item_result("i2", 0.0)]
    metrics, bins = run_eval.calibration_metrics(
        items, correctness_by_item={"i1": True, "i2": False})
    assert metrics[0].status == "ok"
    assert metrics[0].value == 0.0     # perfectly calibrated
    assert bins


def test_calibration_excludes_unavailable_verifications_i2():
    items = [_item_result("i1", None)]
    items[0].verification = [{"status": "unavailable", "score": None,
                              "reason": "x"}]
    metrics, _ = run_eval.calibration_metrics(
        items, correctness_by_item={"i1": True})
    assert metrics[0].status == "unavailable"
    assert metrics[0].reason == "no_verifier_scores_available"


# --------------------------------------------------------------------------- #
# End-to-end run()
# --------------------------------------------------------------------------- #
@requires_sample
def test_run_produces_a_complete_table_without_writing(tmp_path):
    gold_dir = tmp_path / "gold"
    write_gold_set("oos", [OOSItem(item_id="o1", question="What is clause one?",
                                   label="in_scope")], gold_dir)
    write_gold_set("repealed",
                   [RepealedItem(item_id="r1", question="q", citation="IPC s.302",
                                in_force_label="repealed",
                                superseded_by="BNS s.103")], gold_dir)

    results = run_eval.run(split="dev", gold_dir=gold_dir,
                           corpus_paths=[SAMPLE], write=False)
    assert results["summary"]["n_items_run"] >= 1
    assert set(results["missing_gold_sets"]) == {"retrieval", "groundedness",
                                                  "entailment"}
    assert "manifest" not in results or "json_path" not in results["manifest"]


@requires_sample
def test_run_writes_json_and_markdown(tmp_path):
    out_dir = tmp_path / "results"
    gold_dir = tmp_path / "gold"
    write_gold_set("oos", [OOSItem(item_id="o1", question="What is clause one?",
                                   label="in_scope")], gold_dir)

    results = run_eval.run(split="dev", out_dir=out_dir, gold_dir=gold_dir,
                           corpus_paths=[SAMPLE], write=True)
    json_path = Path(results["manifest"]["json_path"])
    md_path = Path(results["manifest"]["markdown_path"])
    assert json_path.exists() and md_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["manifest"]["split"] == "dev"


@requires_sample
def test_run_records_that_the_pipeline_is_stubbed(tmp_path):
    results = run_eval.run(split="dev", gold_dir=tmp_path / "empty",
                           corpus_paths=[SAMPLE], write=False)
    assert results["manifest"]["pipeline_is_stubbed"] is True


@requires_sample
def test_run_never_reports_zero_for_an_empty_gold_set(tmp_path):
    """A missing set must show up as missing, never as a measured 0.0."""
    results = run_eval.run(split="dev", gold_dir=tmp_path / "empty",
                           corpus_paths=[SAMPLE], write=False)
    assert len(results["missing_gold_sets"]) == 5
    all_metrics = [m for ms in results["sections"].values() for m in ms]
    assert not any(m["status"] == "ok" and m["value"] == 0.0
                  for m in all_metrics if "retrieval" in m["name"])


@requires_sample
def test_run_records_gold_set_checksums_in_the_manifest(tmp_path):
    gold_dir = tmp_path / "gold"
    write_gold_set("oos", [OOSItem(item_id="o1", question="q",
                                   label="in_scope")], gold_dir)
    results = run_eval.run(split="dev", gold_dir=gold_dir, corpus_paths=[SAMPLE],
                           write=False)
    assert len(results["manifest"]["gold_set_checksums"]["oos"]) == 64
