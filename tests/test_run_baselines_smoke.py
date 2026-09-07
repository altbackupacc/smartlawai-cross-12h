"""End-to-end structural smoke test for baselines/run_baselines.py, against
sample_docs/sample_judgment.txt with mocked generators (no real network
calls, no `ml`-extras GPU deps needed for the non-hybrid conditions exercised
here). Since eval/ (M6) is not present in this checkout, the 'metrics' field
is expected to report itself as unavailable rather than compute real numbers
or crash -- see run_baselines.py's _try_score docstring."""
from __future__ import annotations

import os
import tempfile

import pytest


def _backend(tmp):
    os.environ["SMARTLAW_LOCAL_DIR"] = tmp
    from smartlawai.adapters.factory import get_backend
    return get_backend("local")


def _ingest_sample_judgment(be, owner_id="alice"):
    from smartlawai.pipeline import Pipeline
    here = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(here, "sample_docs", "sample_judgment.txt")
    doc_id, _ = Pipeline(be).ingest(path, doc_type="JUDGMENT", source="test",
                                    owner_id=owner_id)
    return doc_id


def test_run_zero_shot_condition_smoke(tmp_path, monkeypatch):
    import baselines.run_baselines as rb
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        doc_id = _ingest_sample_judgment(be)
        scope = Scope(doc_ids=(doc_id,), owner_id="alice")

        monkeypatch.setattr("baselines.generators.mistral_client.complete",
                            lambda *a, **k: "a mocked mistral answer")

        result = rb.run("mistral_zero_shot",
                        [("What is the arbitration clause?", scope)],
                        backend=be, out_dir=tmp_path)

    assert result["condition"] == "mistral_zero_shot"
    assert result["n_items"] == 1
    assert result["items"][0]["model_id"]
    assert result["contamination"] is None  # model_id_for_cutoff not requested
    assert isinstance(result["metrics"], str)
    assert result["metrics"].startswith("unavailable")
    written = list(tmp_path.glob("mistral_zero_shot_*.json"))
    assert len(written) == 1


def test_run_unknown_condition_raises():
    import baselines.run_baselines as rb

    with pytest.raises(ValueError):
        rb.run("not-a-real-condition", [])


def test_run_bm25_frontier_condition_actually_retrieves(tmp_path, monkeypatch):
    """The load-bearing baseline (PLAN.md's 'watch for' line): confirms
    retrieval genuinely ran (non-empty passage_ids) rather than the generator
    silently falling back to zero-shot."""
    import baselines.run_baselines as rb
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        doc_id = _ingest_sample_judgment(be)
        scope = Scope(doc_ids=(doc_id,), owner_id="alice")

        monkeypatch.setattr(
            "baselines.generators.complete_frontier",
            lambda *a, **k: "damages under Section 74 of the Indian Contract Act, 1872")

        result = rb.run("bm25_frontier", [("What about damages?", scope)],
                        backend=be, out_dir=tmp_path)

    assert result["items"][0]["claims"][0]["passage_ids"]


def test_run_with_contamination_split_requested(tmp_path, monkeypatch):
    from datetime import date

    import baselines.run_baselines as rb
    from baselines import config
    from smartlawai.scope import Scope

    monkeypatch.setitem(config.MODEL_CUTOFF_DATES, "test-cutoff-model", date(2030, 1, 1))

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        doc_id = _ingest_sample_judgment(be)
        scope = Scope(doc_ids=(doc_id,), owner_id="alice")

        monkeypatch.setattr("baselines.generators.mistral_client.complete",
                            lambda *a, **k: "an answer")

        result = rb.run("mistral_zero_shot", [("a question", scope)], backend=be,
                        model_id_for_cutoff="test-cutoff-model", out_dir=tmp_path)

    assert result["contamination"] is not None
    all_bucketed = (result["contamination"]["pre_cutoff"]
                    + result["contamination"]["post_cutoff"]
                    + result["contamination"]["undated_skipped"])
    assert doc_id in all_bucketed
