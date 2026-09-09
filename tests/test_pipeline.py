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


def test_ask_small_to_big_returns_full_parent_clauses():
    """Verify that retrieval matches child chunks but returns full parent clauses to generation."""
    from smartlawai.pipeline import Pipeline
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        # Create a document with a long clause that produces children
        long_clause = (
            "Section 27. Agreement in restraint of trade, void. "
            "Every agreement by which anyone is restrained from exercising a lawful "
            "profession, trade or business of any kind, is to that extent void. "
            "Exception 1 -- Saving of agreement not to carry on business of which good-will is sold. "
            "One who sells the good-will of a business may agree with the buyer to refrain from "
            "carrying on a similar business, within specified local limits, so long as the buyer, "
            "or any person deriving title to the good-will from him, carries on a like business therein, "
            "provided that such limits appear to the Court reasonable, regard being had to the nature of the business."
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(long_clause)
            path = f.name

        pipeline = Pipeline(be)
        doc_id, _ = pipeline.ingest(path, doc_type="STATUTE", source="bare_act", owner_id="alice")

        chunks = be.fetch_chunks(doc_id)
        parent_chunks = [c for c in chunks if c.parent_chunk_id is None]
        child_chunks = [c for c in chunks if c.parent_chunk_id is not None]
        assert len(parent_chunks) == 1
        assert len(child_chunks) >= 2

        # Ask a query targeting the clause
        res = pipeline.ask("Is agreement in restraint of trade void?", Scope(doc_ids=(doc_id,), owner_id="alice"))

        # In small-to-big retrieval, top_chunk_ids must resolve to the parent chunk!
        top_ids = res.trace.retrieval["top_chunk_ids"]
        assert parent_chunks[0].chunk_id in top_ids
        # Child chunk ids should not be in top_chunk_ids
        for c in child_chunks:
            assert c.chunk_id not in top_ids

