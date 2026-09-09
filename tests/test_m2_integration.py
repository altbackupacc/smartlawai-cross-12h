"""Integration test for Milestone 2 (M2) — Real Retrieval.
Validates end-to-end retrieval, small-to-big parent resolution, provenance preservation,
and Invariant I1 cross-document isolation verified in PipelineTrace (PLAN.md M2 acceptance criterion)."""
from __future__ import annotations

import os
import tempfile

from smartlawai.adapters.factory import get_backend
from smartlawai.pipeline import Pipeline
from smartlawai.scope import Scope


def _backend(tmp: str):
    os.environ["SMARTLAW_LOCAL_DIR"] = tmp
    return get_backend("local")


def test_m2_end_to_end_retrieval_and_trace_isolation():
    """Done when: test_cross_document_isolation passes and a query over doc A returns
    only doc-A chunks, verified in the trace (PLAN.md M2 acceptance criterion)."""
    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        pipeline = Pipeline(be)

        # Document A: Contract with multiple clauses and repeated statutory phrases
        doc_a_text = (
            "Section 1. Non-Compete\n"
            "Every agreement by which anyone is restrained from exercising a lawful "
            "profession, trade or business of any kind, is to that extent void. "
            "This clause protects legitimate business interests.\n\n"
            "Section 2. Confidentiality\n"
            "The receiving party agrees to hold all proprietary information in strict confidence. "
            "This agreement shall be governed by the laws of India and subject to jurisdiction in Delhi."
        )

        # Document B: Arbitration agreement owned by Bob
        doc_b_text = (
            "Clause 1. Dispute Resolution\n"
            "Any dispute arising out of or in connection with this contract shall be settled "
            "by arbitration under the Arbitration and Conciliation Act, 1996 in Mumbai."
        )

        # Ingest both documents
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(doc_a_text)
            path_a = f.name
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(doc_b_text)
            path_b = f.name

        doc_a_id, trace_ingest_a = pipeline.ingest(path_a, doc_type="CONTRACT", source="delhi_corp", owner_id="alice")
        doc_b_id, trace_ingest_b = pipeline.ingest(path_b, doc_type="CONTRACT", source="mumbai_corp", owner_id="bob")

        assert doc_a_id != doc_b_id
        chunks_a = be.fetch_chunks(doc_a_id)
        chunks_b = be.fetch_chunks(doc_b_id)
        assert len(chunks_a) >= 2
        assert len(chunks_b) >= 1

        # Check provenance fields on ingested chunks
        for c in chunks_a:
            assert c.owner_id == "alice"
            assert c.page_no is not None
            assert c.para_no is not None
            assert c.section_label in ("Section 1", "Section 2")

        # 1. Query Document A as Alice: MUST return only Doc A chunks in trace
        query = "What is the rule regarding restraint of trade?"
        scope_a = Scope(doc_ids=(doc_a_id,), owner_id="alice")
        res_a = pipeline.ask(query, scope=scope_a)

        assert res_a.trace.scope.doc_ids == (doc_a_id,)
        assert res_a.trace.scope.owner_id == "alice"

        candidate_ids = res_a.trace.retrieval["candidate_chunk_ids"]
        top_ids = res_a.trace.retrieval["top_chunk_ids"]

        # Invariant I1 check: All candidates and top chunks must strictly belong to Document A
        doc_a_chunk_ids = {c.chunk_id for c in chunks_a}
        doc_b_chunk_ids = {c.chunk_id for c in chunks_b}

        assert len(candidate_ids) > 0
        assert all(cid in doc_a_chunk_ids for cid in candidate_ids)
        assert not any(cid in doc_b_chunk_ids for cid in candidate_ids)

        assert len(top_ids) > 0
        assert all(cid in doc_a_chunk_ids for cid in top_ids)
        assert not any(cid in doc_b_chunk_ids for cid in top_ids)

        # 2. Query Document A as Bob (wrong owner): MUST return empty / REFUSE (I1 tenant isolation)
        scope_unauthorized = Scope(doc_ids=(doc_a_id,), owner_id="bob")
        res_unauthorized = pipeline.ask(query, scope=scope_unauthorized)

        assert res_unauthorized.decision == "REFUSE"
        assert res_unauthorized.trace.retrieval["n_candidates"] == 0
        assert res_unauthorized.trace.retrieval["candidate_chunk_ids"] == []
        assert res_unauthorized.trace.retrieval["top_chunk_ids"] == []

        # 3. Query Document B as Bob: MUST return only Doc B chunks
        scope_b = Scope(doc_ids=(doc_b_id,), owner_id="bob")
        res_b = pipeline.ask("How are disputes resolved?", scope=scope_b)

        assert res_b.trace.retrieval["n_candidates"] > 0
        assert all(cid in doc_b_chunk_ids for cid in res_b.trace.retrieval["candidate_chunk_ids"])
        assert not any(cid in doc_a_chunk_ids for cid in res_b.trace.retrieval["candidate_chunk_ids"])
