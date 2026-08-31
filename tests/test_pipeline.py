"""End-to-end M0 skeleton test: ingest -> ask, asserting every stage appears in
the trace (PLAN.md M0: "a stage missing from the trace here will be missing
forever")."""
from __future__ import annotations

import os
import tempfile


def _backend(tmp):
    os.environ["SMARTLAW_LOCAL_DIR"] = tmp
    from smartlawai.adapters.factory import get_backend
    return get_backend("local")


def test_ingest_trace_covers_every_stage():
    from smartlawai.pipeline import Pipeline

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("1. First clause. WHEREAS the parties agree. 2. Second clause.")
            path = f.name

        doc_id, trace = Pipeline(be).ingest(path, doc_type="CONTRACT",
                                            source="test", owner_id="alice")
        stage_names = {s.name for s in trace.stages}
        assert stage_names == {"ingest", "chunk"}
        assert all(s.status == "ok" for s in trace.stages)
        assert doc_id.startswith("doc-")


def test_ask_trace_covers_every_stage_and_scopes_correctly():
    from smartlawai.pipeline import Pipeline
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("1. First clause. WHEREAS the parties agree. 2. Second clause.")
            path = f.name

        pipeline = Pipeline(be)
        doc_id, _ = pipeline.ingest(path, doc_type="CONTRACT", source="test",
                                    owner_id="alice")

        result = pipeline.ask("What does the first clause say?",
                              Scope(doc_ids=(doc_id,), owner_id="alice"))

        stage_names = {s.name for s in result.trace.stages}
        assert stage_names == {"retrieve", "rerank", "generate", "verify", "decide"}
        assert result.decision in ("ANSWER", "REFUSE")
        assert result.trace.retrieval["n_candidates"] > 0


def test_ask_refuses_when_scope_has_no_chunks():
    from smartlawai.pipeline import Pipeline
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        result = Pipeline(be).ask(
            "any question",
            Scope(doc_ids=("doc-nonexistent",), owner_id="alice"))
        assert result.decision == "REFUSE"
        assert result.trace.decision["reason"] == "no_claims"
