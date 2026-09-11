"""THE single pipeline. api/main.py and cli.py both call this; a later
research/eval harness (M6) calls the same class, unmodified. All models are
stubbed here (PLAN.md M0) -- M2/M3/M5 pass real Encoder/Reranker/Generator/
Verifier implementations into Pipeline's constructor without touching the
stage sequence below."""
from __future__ import annotations

import logging
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
from smartlawai.verify.entailment import ClaimEntailmentResult, DualEntailmentVerifier

logger = logging.getLogger(__name__)


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
        entailment_verifier: DualEntailmentVerifier | None = None,
    ) -> None:
        self.be = backend
        self.encoder = encoder or StubEncoder()
        self.reranker = reranker or StubReranker()
        self.generator = generator or StubGenerator()
        self.verifier = verifier or StubVerifier()
        # M5 real dual-scorer entailment (HHEM-2.1 + InLegalNLI slot). Optional:
        # when absent, _decide bridges the legacy `verifier` score into gate.py's
        # shape instead (M0 stub default, unit tests) -- see _decide below.
        self.entailment_verifier = entailment_verifier

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
            searchable_candidates = [c for c in candidates if c.bm25_indexed] or candidates
            self.encoder.embed([question])  # proves the seam; device-aware encoding in M2
            rec.detail = {"n_candidates": len(candidates), "n_searchable": len(searchable_candidates)}
        trace.retrieval = {"n_candidates": len(candidates),
                           "candidate_chunk_ids": [c.chunk_id for c in candidates]}

        with trace.stage("rerank") as rec:
            ranked = self.reranker.rerank(question, searchable_candidates, config.RETRIEVE_TOP_K)
            # Small-to-big retrieval: resolve child chunks to full parent clauses,
            # deduplicating parents to prevent near-duplicate crowding (PLAN.md M2)
            by_id = {c.chunk_id: c for c in candidates}
            seen_parents: set[str] = set()
            top: list[RetrievedChunk] = []
            for rc in ranked:
                p_id = rc.chunk.parent_chunk_id or rc.chunk.chunk_id
                if p_id in seen_parents:
                    continue
                seen_parents.add(p_id)
                parent_chunk = by_id.get(p_id)
                if not parent_chunk:
                    fetched = self.be.fetch_chunks_by_ids([p_id])
                    parent_chunk = fetched[0] if fetched else rc.chunk
                top.append(RetrievedChunk(chunk=parent_chunk, score=rc.score))
                if len(top) >= config.RERANK_TOP_K:
                    break
            rec.detail = {"n_top": len(top)}
        trace.retrieval["top_chunk_ids"] = [rc.chunk.chunk_id for rc in top]

        with trace.stage("generate") as rec:
            gen = self.generator.generate(question, top)
            rec.detail = {
                "n_claims": len(gen.claims),
                "model": gen.model_id,
                "n_unanswerable": len(gen.unanswerable_aspects),
            }
        trace.generation = {
            "model": gen.model_id,
            "n_claims": len(gen.claims),
            "claims": [
                {
                    "text": c.text,
                    "passage_ids": c.passage_ids,
                    "citations": c.citations,
                }
                for c in gen.claims
            ],
            "unanswerable_aspects": gen.unanswerable_aspects,
        }

        with trace.stage("verify") as rec:
            results = [self.verifier.verify(cl, top) for cl in gen.claims]
            entailment_results: list[ClaimEntailmentResult] | None = None
            if self.entailment_verifier is not None:
                passages_by_id = {rc.chunk.chunk_id: rc for rc in top}
                entailment_results = [
                    self.entailment_verifier.verify_claim(cl, passages_by_id)
                    for cl in gen.claims
                ]
            rec.detail = {"n_verified": len(results)}
        trace.verification = {"results": [
            {"status": r.status, "score": r.score, "reason": r.reason} for r in results]}
        if entailment_results is not None:
            trace.verification["entailment"] = [
                {
                    "hhem_status": r.hhem_score.status,
                    "hhem_value": r.hhem_score.value,
                    "inlegalnli_status": r.inlegalnli_score.status,
                    "is_entailed": r.is_entailed,
                    "status": r.status,
                    "reason": r.reason,
                }
                for r in entailment_results
            ]

        with trace.stage("decide") as rec:
            outcome, reason = self._decide(
                results, gen.claims, passages=top, entailment_results=entailment_results
            )
            rec.detail = {"outcome": outcome, "reason": reason}
        trace.decision = {"outcome": outcome, "reason": reason}

        # Estimate token usage (~4 characters per token) and record cost
        prompt_chars = len(question) + sum(len(rc.chunk.chunk_text) for rc in top)
        gen_chars = sum(len(c.text) for c in gen.claims)
        generator_tokens = (prompt_chars + gen_chars) // 4
        # Mistral-7B L4 / vLLM cost model ~$0.0002 per 1k tokens
        estimated_usd = round(generator_tokens * 0.0000002, 6)
        trace.cost = {"generator_tokens": generator_tokens, "estimated_usd": estimated_usd}

        return AnswerResult(answer=self._render(gen, outcome), decision=outcome, trace=trace)

    def _decide(
        self,
        results: list[VerificationResult],
        claims: list[Claim],
        passages: list[RetrievedChunk] | None = None,
        entailment_results: list[ClaimEntailmentResult] | None = None,
    ) -> tuple[str, str]:
        """Deterministic application code (I3) using M5 gate conjunction."""
        if not claims:
            return "REFUSE", "no_claims"
        if any(r.status == "unavailable" for r in results):
            return "REFUSE", "verifier_unavailable"

        from smartlawai.gate import GateOutcome, decide_gate
        from smartlawai.verify.structural import verify_structural

        struct_results = verify_structural(claims, passages or [])

        if entailment_results is None:
            # No real DualEntailmentVerifier injected (M0 stub default / unit
            # tests): bridge the generic verifier's score into gate.py's
            # ClaimEntailmentResult shape so the gate conjunction still runs.
            from smartlawai.verify.entailment import EntailmentScore

            entailment_results = []
            for cl, vr in zip(claims, results):
                is_ent = (
                    (vr.score is not None and vr.score >= config.FAITHFULNESS_THRESHOLD)
                    if vr.status == "ok"
                    else False
                )
                entailment_results.append(
                    ClaimEntailmentResult(
                        claim_text=cl.text,
                        passage_id=cl.passage_ids[0] if cl.passage_ids else "",
                        hhem_score=EntailmentScore(name="verifier", value=vr.score, status=vr.status, reason=vr.reason),
                        inlegalnli_score=EntailmentScore(name="inlegalnli", value=None, status="unavailable", reason="m8a_slot"),
                        is_entailed=is_ent,
                        status=vr.status,
                        reason=vr.reason,
                    )
                )

        decision = decide_gate(claims, struct_results, entailment_results)
        if decision.outcome == GateOutcome.ALLOW:
            return "ANSWER", decision.reason
        elif decision.outcome == GateOutcome.PARTIAL_ALLOW:
            return "PARTIAL_ALLOW", decision.reason
        else:
            return "REFUSE", decision.reason

    def _render(self, gen: GenerationResult, outcome: str) -> str:
        if outcome == "REFUSE":
            return "I cannot verify an answer from the available document(s)."
        return " ".join(c.text for c in gen.claims)


def build_production_pipeline(backend: BackendInterface, device: str | None = None) -> Pipeline:
    """Composition root wiring the real M2/M3/M5 components into serving
    (api/main.py, ui/app.py) instead of the M0 stub defaults.

    `encoder`/`reranker` load model weights at construction time with no
    built-in fallback, so each is attempted independently and, on failure
    (weights not cached locally and no network -- exactly the "no model
    downloads in the agent loop" situation CLAUDE.md #6 describes), this
    falls back to that one stub rather than refusing to start the app. The
    generator and entailment verifier are safe to always construct: both are
    fail-closed at call time (I2) rather than at construction -- an
    unreachable Mistral endpoint or unloadable HHEM model degrades to REFUSE,
    it never crashes serving or fabricates an answer.
    """
    encoder: Encoder | None = None
    reranker: Reranker | None = None

    try:
        from smartlawai.core.inlegalbert import InLegalBERTEncoder

        encoder = InLegalBERTEncoder(device=device)
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_production_pipeline: falling back to StubEncoder: %s", exc)

    try:
        from smartlawai.core.rerank import Reranker as RealReranker

        reranker = RealReranker(device=device)
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_production_pipeline: falling back to StubReranker: %s", exc)

    from smartlawai.core.generate import StructuredMistralGenerator
    from smartlawai.verify.entailment import HHEMScorer

    return Pipeline(
        backend,
        encoder=encoder,
        reranker=reranker,
        generator=StructuredMistralGenerator(),
        entailment_verifier=DualEntailmentVerifier(hhem_scorer=HHEMScorer(device=device)),
    )


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
