"""End-to-end M6 integration (Phase 13) -- the literal done-when criterion.

`M6_ONBOARDING.md` §2: "the full table computes for the untrained system. That
is your floor." `M6_ONBOARDING.md` §17 step 11 names this as the final check.

Also enforces the layering rule stated in `eval/__init__.py` and assumed by
`OPS.md` §8 ("the eval harness is pure functions over data structures. Neither
imports the pipeline."): every M6 module except `run_eval.py` (and the app's
optional smoke-input path) must import nothing from `smartlawai`.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from eval import config, run_eval
from eval.gold_sets import load_all_gold_sets
from eval.schemas import GOLD_SET_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
SAMPLE = REPO_ROOT / "sample_docs" / "sample_judgment.txt"
requires_sample = pytest.mark.skipif(not SAMPLE.exists(),
                                     reason="sample_docs/sample_judgment.txt missing")

# run_eval.py is the one module the onboarding requires to call Pipeline
# directly. app.py may import smartlawai only inside its own guarded main path
# for smoke input; it is exempted from the static check and instead checked by
# its own test suite never invoking that path without Streamlit installed.
LAYERING_EXEMPT = {"run_eval.py"}


def _imports_smartlawai(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(n.name.split(".")[0] == "smartlawai" for n in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "smartlawai":
            return True
    return False


def test_metric_layer_does_not_import_smartlawai():
    """OPS.md §8: the eval harness is pure functions over data structures."""
    offenders = []
    for path in EVAL_DIR.rglob("*.py"):
        if path.name in LAYERING_EXEMPT or "annotate" + "/app.py" in str(path).replace(
                "\\", "/"):
            continue
        if _imports_smartlawai(path):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"eval modules importing smartlawai: {offenders}"


def test_only_run_eval_imports_the_pipeline_class():
    """A second import site would mean the pipeline's stage sequence is at risk
    of being reimplemented somewhere in eval/, which the onboarding forbids."""
    hits = []
    for path in EVAL_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bimport\s+Pipeline\b|from\s+smartlawai\.pipeline\s+import",
                     text):
            hits.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
    assert hits == ["eval/run_eval.py"]


def test_old_src_eval_package_is_gone():
    """M6_ONBOARDING.md §4.3: don't leave a duplicate."""
    assert not (REPO_ROOT / "src" / "smartlawai" / "eval").exists()


def test_top_level_eval_package_is_where_metrics_now_live():
    assert (EVAL_DIR / "metrics.py").exists()
    assert (EVAL_DIR / "metrics.py").read_text(encoding="utf-8").count(
        "def rouge_l(") == 0, "hand-rolled rouge_l must be deleted (PLAN.md M6)"


# --------------------------------------------------------------------------- #
# The done-when criterion
# --------------------------------------------------------------------------- #
@requires_sample
def test_full_table_computes_end_to_end_on_the_untrained_system(tmp_path):
    """The literal M6_ONBOARDING.md §2 done-when: run_eval over every gold set
    the repo ships, on the real (stubbed) Pipeline, no crash, a written report."""
    results = run_eval.run(split="dev", out_dir=tmp_path, corpus_paths=[SAMPLE],
                           write=True)

    assert Path(results["manifest"]["json_path"]).exists()
    assert Path(results["manifest"]["markdown_path"]).exists()
    assert results["manifest"]["pipeline_is_stubbed"] is True
    assert results["summary"]["n_items_run"] > 0
    # "It doesn't need to look good. It needs to run and be honest": both
    # measured and unavailable metrics are expected to appear.
    assert results["summary"]["n_metrics_computed"] > 0
    assert results["failures"] == []


@requires_sample
def test_every_shipped_gold_set_participates_in_a_real_run(tmp_path):
    results = run_eval.run(split="dev", out_dir=tmp_path, corpus_paths=[SAMPLE],
                           write=False)
    assert set(results["gold_sets"]) == set(GOLD_SET_NAMES)
    assert results["missing_gold_sets"] == {}


@requires_sample
def test_i1_holds_every_ask_carries_an_explicit_scope():
    """Static proof that the harness cannot bypass Scope: EvalCorpus.scope_for
    is the only path _ask uses to reach Pipeline.ask, and Scope itself raises on
    empty doc_ids (I1) -- exercised directly here rather than by inspection."""
    from smartlawai.scope import Scope
    with pytest.raises(ValueError):
        Scope(doc_ids=(), owner_id=config.EVAL_OWNER_ID)

    with run_eval.EvalCorpus([SAMPLE]) as corpus:
        result = run_eval._ask(corpus, "i1", "oos", "What does clause one say?", [])
        assert result.trace_id  # a trace exists, meaning ask() ran under a Scope


@requires_sample
def test_i2_holds_no_unavailable_component_becomes_a_number(tmp_path):
    """The report's own 'Not measured' bookkeeping IS the I2 check: every
    metric with status != ok must carry value=None, never a substituted 0.0 or
    0.5 -- the exact bug CLAUDE.md I2 names."""
    results = run_eval.run(split="dev", out_dir=tmp_path, corpus_paths=[SAMPLE],
                           write=False)
    for metrics in results["sections"].values():
        for m in metrics:
            if m["status"] != "ok":
                assert m["value"] is None, f"{m['name']} is {m['status']} but " \
                    f"carries value={m['value']!r}"


@requires_sample
def test_run_is_reproducible_given_the_same_gold_data_and_seed(tmp_path):
    """Stub models are deterministic; the harness must not introduce
    nondeterminism of its own (item ordering, dict iteration, etc.).

    Latency numbers are excluded on purpose -- they are real wall-clock
    measurements of THIS machine at THIS moment, and legitimately jitter
    between runs. Reproducibility is a property of the computed metrics over
    fixed inputs, not of timing.
    """
    r1 = run_eval.run(split="dev", out_dir=tmp_path / "a", corpus_paths=[SAMPLE],
                      seed=42, write=False)
    r2 = run_eval.run(split="dev", out_dir=tmp_path / "b", corpus_paths=[SAMPLE],
                      seed=42, write=False)
    from eval.report import flat_metrics

    def non_timing(flat):
        return {k: v for k, v in flat.items() if not k.startswith("latency_")}

    assert non_timing(flat_metrics(r1)) == non_timing(flat_metrics(r2))


@requires_sample
def test_run_over_the_repos_own_shipped_gold_sets_is_the_final_smoke_test():
    """Run against eval/gold/ itself (the seed sets), not a fixture -- proves
    the shipped data and the harness actually agree with each other."""
    loaded, missing = load_all_gold_sets()
    assert not missing
    assert all(loaded[name] for name in GOLD_SET_NAMES)

    with run_eval.EvalCorpus([SAMPLE]) as corpus:
        _, retrieval_metrics = run_eval.run_retrieval(corpus, loaded["retrieval"])
        assert len(retrieval_metrics) == 8  # full RESEARCH.md §5 retrieval row

        answer_metrics = run_eval.score_groundedness(corpus, loaded["groundedness"])
        by_name = {m.name: m for m in answer_metrics}
        assert by_name["groundedness.groundedness_rate"].status == "ok"

        # Audit fix (docs/M6_GAP_ANALYSIS.md G2): these correctly report
        # unavailable -- there is no registry (M4 not started) to resolve
        # citations against, so a real number here would be fabricated.
        authority_metrics = run_eval.score_repealed(loaded["repealed"])
        assert all(m.status == "unavailable" for m in authority_metrics)
