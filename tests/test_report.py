"""Report rendering tests (M6 Phase 12).

Reference image 2 lists "evaluation report (tables + results)" as a first-class
deliverable. The one rule that shapes these tests: a metric that was not
measured must be visibly different from one that measured zero, in both the
JSON and the Markdown.
"""
from __future__ import annotations

import json

from eval import report
from eval.metrics import MetricResult


def _results(**overrides):
    base = {
        "manifest": {"split": "dev", "seed": 42, "generated_at": "t",
                    "harness_version": "1.0.0", "pipeline_is_stubbed": True,
                    "git_commit": "abc123"},
        "gold_sets": {"oos": {"n": 32, "target_n": 350, "sha256": "a" * 64,
                              "duplication_rate_achieved": 0.0,
                              "duplication_rate_target": 0.30,
                              "ci_halfwidth_at_p50": 0.17, "mean_agreement": None}},
        "missing_gold_sets": {"retrieval": "not collected yet"},
        "sizing_notes": ["oos: n=32 (BELOW target, target 350) ..."],
        "sections": {
            "retrieval": [MetricResult.unavailable("recall@5",
                                                    "no_relevance_judgments").to_dict()],
            "answer": [MetricResult("groundedness.groundedness_rate", 0.5, n=10).to_dict()],
            "citation": [], "robustness": [], "calibration": [], "cost": [],
            "human": [],
        },
        "reliability_diagram": [],
        "items": [],
        "failures": [],
        "error_analysis": {},
        "summary": {"n_items_run": 10, "n_failures": 0},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_markdown_renders_without_error():
    md = report.render_markdown(_results())
    assert "# SmartLawAI evaluation report" in md
    assert "`dev`" in md


def test_markdown_flags_a_stubbed_pipeline():
    md = report.render_markdown(_results())
    assert "stub" in md.lower()
    assert "M6 done-when criterion" in md


def test_markdown_does_not_flag_a_stubbed_pipeline_when_real():
    results = _results()
    results["manifest"]["pipeline_is_stubbed"] = False
    md = report.render_markdown(results)
    assert "done-when criterion" not in md


def test_unavailable_metric_renders_as_the_sentinel_not_a_number():
    md = report.render_markdown(_results())
    assert "recall@5" in md
    lines = [line for line in md.splitlines() if "recall@5" in line]
    assert any(report.UNAVAILABLE in line for line in lines)
    assert not any("| 0.0000 |" in line for line in lines)


def test_measured_metric_renders_as_a_number():
    md = report.render_markdown(_results())
    assert "0.5000" in md


def test_not_measured_section_lists_every_unavailable_metric():
    md = report.render_markdown(_results())
    assert "## Not measured" in md
    assert "no_relevance_judgments" in md


def test_not_measured_section_says_so_when_everything_computed():
    results = _results()
    results["sections"]["retrieval"] = [MetricResult("recall@5", 0.9, n=10).to_dict()]
    md = report.render_markdown(results)
    assert "Every applicable metric computed." in md


def test_missing_gold_sets_are_listed_with_their_reason():
    md = report.render_markdown(_results())
    assert "not collected yet" in md
    assert "retrieval" in md.split("## Gold sets not available")[1][:200]


def test_no_failures_renders_a_clean_message():
    md = report.render_markdown(_results())
    assert "No item raised an error" in md


def test_failures_are_tabulated():
    results = _results(failures=[{"item_id": "i1", "stage": "oos",
                                  "error": "boom"}])
    md = report.render_markdown(results)
    assert "`i1`" in md and "boom" in md


def test_reliability_diagram_renders_when_present():
    results = _results(reliability_diagram=[
        {"bin_lower": 0.9, "bin_upper": 1.0, "mean_confidence": 0.95,
         "accuracy": 0.9, "count": 5}])
    md = report.render_markdown(results)
    assert "Reliability diagram data" in md


def test_error_analysis_renders_grouped_counts():
    results = _results(error_analysis={"decisions": {"ANSWER": 8, "REFUSE": 2}})
    md = report.render_markdown(results)
    assert "ANSWER" in md and "8" in md


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def test_write_report_produces_matching_json_and_markdown(tmp_path):
    results = _results()
    json_path, md_path = report.write_report(results, tmp_path, "dev_20260101")
    assert json_path.exists() and md_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["manifest"]["split"] == "dev"
    assert "# SmartLawAI evaluation report" in md_path.read_text(encoding="utf-8")


def test_write_report_creates_the_output_directory(tmp_path):
    out_dir = tmp_path / "nested" / "results"
    report.write_report(_results(), out_dir, "dev_x")
    assert out_dir.exists()


# --------------------------------------------------------------------------- #
# flat_metrics / regression comparison -- what the CI gate uses
# --------------------------------------------------------------------------- #
def test_flat_metrics_includes_only_ok_metrics_with_a_value():
    flat = report.flat_metrics(_results())
    assert flat == {"groundedness.groundedness_rate": 0.5}
    assert "recall@5" not in flat


def test_compare_to_baseline_ignores_latency_jitter():
    """Audit finding: wall-clock latency legitimately jitters between runs on
    the same machine and must not fail the CI gate on that basis alone
    (docs/M6_GAP_ANALYSIS.md, discovered while verifying the audit's fixes)."""
    baseline = _results()
    baseline["sections"]["cost"] = [
        MetricResult("latency_p50_ms[retrieve]", 0.78, n=10).to_dict()]
    current = _results()
    current["sections"]["cost"] = [
        MetricResult("latency_p50_ms[retrieve]", 0.10, n=10).to_dict()]  # big "drop"
    regressions, disappeared = report.compare_to_baseline(current, baseline, 0.05)
    assert regressions == [] and disappeared == []


def test_compare_to_baseline_detects_a_regression():
    baseline = _results()
    baseline["sections"]["answer"] = [
        MetricResult("groundedness.groundedness_rate", 0.9, n=10).to_dict()]
    current = _results()  # 0.5, a drop of 0.4
    regressions, disappeared = report.compare_to_baseline(current, baseline, 0.05)
    assert regressions and regressions[0]["metric"] == "groundedness.groundedness_rate"
    assert regressions[0]["delta"] == -0.4
    assert disappeared == []


def test_compare_to_baseline_within_tolerance_is_not_a_regression():
    baseline = _results()
    baseline["sections"]["answer"] = [
        MetricResult("groundedness.groundedness_rate", 0.52, n=10).to_dict()]
    regressions, _ = report.compare_to_baseline(_results(), baseline, 0.10)
    assert regressions == []


def test_compare_to_baseline_flags_a_metric_that_disappeared():
    """A harness that stops computing a metric must not pass silently."""
    baseline = _results()
    baseline["sections"]["citation"] = [
        MetricResult("citation_validity_rate", 0.8, n=5).to_dict()]
    current = _results()  # citation_validity_rate absent
    _regressions, disappeared = report.compare_to_baseline(current, baseline, 0.05)
    assert disappeared == [{"metric": "citation_validity_rate", "baseline": 0.8}]


def test_compare_to_baseline_ignores_a_metric_that_newly_appeared():
    baseline = _results()
    current = _results()
    current["sections"]["citation"] = [
        MetricResult("citation_validity_rate", 0.8, n=5).to_dict()]
    regressions, disappeared = report.compare_to_baseline(current, baseline, 0.05)
    assert regressions == [] and disappeared == []


def test_as_dicts_accepts_metricresult_objects_or_plain_dicts():
    mixed = [MetricResult("a", 1.0, n=1), {"name": "b", "value": 2.0, "n": 1,
                                           "status": "ok", "reason": ""}]
    out = report.as_dicts(mixed)
    assert out[0]["name"] == "a" and out[1]["name"] == "b"
