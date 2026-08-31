"""THE single pipeline. api/main.py and cli.py both call this; a later
research/eval harness (M6) calls the same class, unmodified. All models are
stubbed here (PLAN.md M0) -- M2/M3/M5 pass real Encoder/Reranker/Generator/
Verifier implementations into Pipeline's constructor without touching the
stage sequence below."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from smartlawai import config
from smartlawai.adapters.base import BackendInterface, RetrievedChunk
from smartlawai.core.chunking import chunk_document
from smartlawai.core.ocr import ingest_file
from smartlawai.core.preprocess import preprocess_doc
from smartlawai.protocols import (
    Claim,
    Encoder,
    GenerationResult,
    Generator,
    Reranker,
    VerificationResult,
    Verifier,
)
from smartlawai.scope import Scope
from smartlawai.trace import PipelineTrace


@dataclass
class AnswerResult:
    answer: str
    decision: str  # "ANSWER" | "REFUSE"
    trace: PipelineTrace


class Pipeline:
    def __init__(
        self,
        backend: BackendInterface,
        encoder: Encoder | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self.be = backend
        self.encoder = encoder or StubEncoder()
        self.reranker = reranker or StubReranker()
        self.generator = generator or StubGenerator()
        self.verifier = verifier or StubVerifier()

    # ---- write path: no Scope (a doc_id doesn't exist until ingest completes) ----
    def ingest(self, path: str, doc_type: str, source: str,
               owner_id: str) -> tuple[str, PipelineTrace]:
        trace = PipelineTrace(trace_id=f"tr-{uuid.uuid4().hex[:10]}", scope=None)

        with trace.stage("ingest") as rec:
            res = ingest_file(path, self.be, doc_type=doc_type, source=source,
                               owner_id=owner_id)
            rec.detail = {"doc_id": res.doc_id, "ocr_status": res.ocr_status,
                          "pages": res.num_pages, "language": res.lang_detected}
            if res.ocr_status == "OCR_FAIL":
                rec.status, rec.error = "error", res.error or "ocr_failed"

        n_chunks = 0
        if res.ocr_status != "OCR_FAIL":
            with trace.stage("chunk") as rec:
                cleaned = preprocess_doc(res.doc_id, res.text, self.be)
                chunks = chunk_document(res.doc_id, cleaned)
                for c in chunks:
                    c.owner_id = owner_id
                self.be.store_chunks(chunks)
                n_chunks = len(chunks)
                rec.detail = {"n_chunks": n_chunks}

        trace.cost = {"ocr_pages": res.num_pages, "estimated_usd": 0.0}
        return res.doc_id, trace

    # ---- read path: Scope required, no default (I1) ----
    def ask(self, question: str, scope: Scope) -> AnswerResult:
        trace = PipelineTrace(trace_id=f"tr-{uuid.uuid4().hex[:10]}", scope=scope)

        with trace.stage("retrieve") as rec:
            candidates = self.be.fetch_chunks_scoped(scope)
            self.encoder.embed([question])  # proves the seam; unused for ranking in M0
            rec.detail = {"n_candidates": len(candidates)}
        trace.retrieval = {"n_candidates": len(candidates),
                           "candidate_chunk_ids": [c.chunk_id for c in candidates]}

        with trace.stage("rerank") as rec:
            top = self.reranker.rerank(question, candidates, config.RERANK_TOP_K)
            rec.detail = {"n_top": len(top)}
        trace.retrieval["top_chunk_ids"] = [rc.chunk.chunk_id for rc in top]

        with trace.stage("generate") as rec:
            gen = self.generator.generate(question, top)
            rec.detail = {"n_claims": len(gen.claims), "model": gen.model_id}
        trace.generation = {"model": gen.model_id, "n_claims": len(gen.claims),
                            "unanswerable_aspects": gen.unanswerable_aspects}

        with trace.stage("verify") as rec:
            results = [self.verifier.verify(cl, top) for cl in gen.claims]
            rec.detail = {"n_verified": len(results)}
        trace.verification = {"results": [
            {"status": r.status, "score": r.score, "reason": r.reason} for r in results]}

        with trace.stage("decide") as rec:
            outcome, reason = self._decide(results, gen.claims)
            rec.detail = {"outcome": outcome, "reason": reason}
        trace.decision = {"outcome": outcome, "reason": reason}
        trace.cost = {"generator_tokens": 0, "estimated_usd": 0.0}

        return AnswerResult(answer=self._render(gen, outcome), decision=outcome, trace=trace)

    def _decide(self, results: list[VerificationResult], claims: list[Claim]) -> tuple[str, str]:
        """Deterministic application code (I3). Kept as a single small method so
        M5's gate.py can replace just its body with the real structural AND
        entailment AND registry conjunction -- stage sequence and call sites
        don't change."""
        if not claims:
            return "REFUSE", "no_claims"
        if any(r.status == "unavailable" for r in results):
            return "REFUSE", "verifier_unavailable"
        return "ANSWER", "ok"

    def _render(self, gen: GenerationResult, outcome: str) -> str:
        if outcome == "REFUSE":
            return "I cannot verify an answer from the available document(s)."
        return " ".join(c.text for c in gen.claims)


class StubEncoder:
    def embed(self, texts: list[str]):
        import numpy as np
        return np.zeros((len(texts), 8), dtype="float32")


class StubReranker:
    def rerank(self, query, candidates, top_k):
        return [RetrievedChunk(chunk=c, score=1.0) for c in candidates[:top_k]]


class StubGenerator:
    def generate(self, question, passages):
        if not passages:
            return GenerationResult(claims=[], unanswerable_aspects=[question],
                                    model_id=config.STUB_GENERATOR_MODEL_ID)
        claim = Claim(text=f"[stub] placeholder answer to: {question}",
                     passage_ids=[p.chunk.chunk_id for p in passages])
        return GenerationResult(claims=[claim], unanswerable_aspects=[],
                                model_id=config.STUB_GENERATOR_MODEL_ID)


class StubVerifier:
    def verify(self, claim, passages):
        if not claim.passage_ids:
            return VerificationResult(status="unavailable", score=None,
                                      reason="claim_has_no_passage_id")
        return VerificationResult(status="ok", score=1.0, reason="stub_pass")
