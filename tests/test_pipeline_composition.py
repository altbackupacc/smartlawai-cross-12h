"""Tests for build_production_pipeline (composition root for api/main.py and
ui/app.py, PLAN.md M3/M5 serving wiring).

These never construct a real InLegalBERTEncoder/Reranker/HHEM model -- doing
so downloads weights, which the agent loop must not do (CLAUDE.md #6). Every
heavy constructor is monkeypatched so the test only exercises
build_production_pipeline's own wiring and fallback logic.
"""

from __future__ import annotations

import tempfile

from smartlawai.core.generate import StructuredMistralGenerator
from smartlawai.pipeline import (
    Pipeline,
    StubEncoder,
    StubReranker,
    build_production_pipeline,
)
from smartlawai.verify.entailment import DualEntailmentVerifier


def _backend(tmp):
    import os

    os.environ["SMARTLAW_LOCAL_DIR"] = tmp
    from smartlawai.adapters.factory import get_backend

    return get_backend("local")


class _FakeRealEncoder:
    def __init__(self, device=None):
        self.device = device

    def embed(self, texts):
        raise NotImplementedError("not exercised in this test")


class _FakeRealReranker:
    def __init__(self, device=None):
        self.device = device

    def rerank(self, query, candidates, top_k):
        raise NotImplementedError("not exercised in this test")


def test_build_production_pipeline_wires_real_generator_and_entailment_verifier(monkeypatch):
    """Regardless of encoder/reranker availability, the generator and
    entailment verifier are always the real M3/M5 components -- both are
    fail-closed at call time (I2), so there's no reason to fall back to a
    stub for either at construction time."""
    monkeypatch.setattr("smartlawai.core.inlegalbert.InLegalBERTEncoder", _FakeRealEncoder)
    monkeypatch.setattr("smartlawai.core.rerank.Reranker", _FakeRealReranker)

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        pipeline = build_production_pipeline(be)

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.encoder, _FakeRealEncoder)
    assert isinstance(pipeline.reranker, _FakeRealReranker)
    assert isinstance(pipeline.generator, StructuredMistralGenerator)
    assert isinstance(pipeline.entailment_verifier, DualEntailmentVerifier)


def test_build_production_pipeline_falls_back_to_stubs_when_models_unavailable(monkeypatch):
    """If InLegalBERT/reranker weights aren't cached and there's no network
    (exactly the "no model downloads in the agent loop" situation), serving
    must not crash -- it degrades to the M0 stub for that component only."""

    def _raise_encoder(*args, **kwargs):
        raise OSError("model weights not cached and no network")

    def _raise_reranker(*args, **kwargs):
        raise OSError("model weights not cached and no network")

    monkeypatch.setattr("smartlawai.core.inlegalbert.InLegalBERTEncoder", _raise_encoder)
    monkeypatch.setattr("smartlawai.core.rerank.Reranker", _raise_reranker)

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        pipeline = build_production_pipeline(be)

    assert isinstance(pipeline.encoder, StubEncoder)
    assert isinstance(pipeline.reranker, StubReranker)
    # Generator/entailment verifier are unaffected by the encoder/reranker failure.
    assert isinstance(pipeline.generator, StructuredMistralGenerator)
    assert isinstance(pipeline.entailment_verifier, DualEntailmentVerifier)


def test_pipeline_entailment_verifier_feeds_gate_directly(monkeypatch):
    """When a real entailment_verifier is injected, Pipeline._decide must use
    its ClaimEntailmentResult directly rather than bridging the generic
    verifier's score -- otherwise wiring a real HHEM verifier in would be a
    no-op."""
    from smartlawai.adapters.base import Chunk, RetrievedChunk
    from smartlawai.protocols import Claim, VerificationResult
    from smartlawai.verify.entailment import EntailmentScore, HHEMScorer

    class _StubHHEM(HHEMScorer):
        def score_pair(self, passage, claim):
            # Always confidently NOT entailed, unlike the generic StubVerifier
            # (pipeline.py's StubVerifier always reports score=1.0/"ok").
            return EntailmentScore(name="hhem-2.1", value=0.01, status="ok", reason="mock_contradiction")

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        pipeline = Pipeline(be, entailment_verifier=DualEntailmentVerifier(hhem_scorer=_StubHHEM()))

        claim = Claim(text="claim", passage_ids=["p1"], citations=[])
        chunk = Chunk(
            chunk_id="p1", doc_id="d1", chunk_index=0, chunk_text="unrelated passage text",
            char_start=0, char_end=10,
        )
        top = [RetrievedChunk(chunk=chunk, score=1.0)]

        outcome, reason = pipeline._decide(
            results=[VerificationResult(status="ok", score=1.0, reason="stub_pass")],
            claims=[claim],
            passages=top,
            entailment_results=[
                pipeline.entailment_verifier.verify_claim(claim, {"p1": top[0]})
            ],
        )

    # The real HHEM mock scored this as not entailed; the gate must REFUSE
    # even though the legacy stub verifier (score=1.0) would have allowed it.
    assert outcome == "REFUSE"
