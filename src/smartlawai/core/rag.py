"""Advisory RAG: HyDE -> hybrid retrieve (BM25 + dense, RRF) -> cross-encoder
rerank -> grounded generate -> faithfulness gate. Refuses out-of-scope or
low-faithfulness answers instead of hallucinating."""
from __future__ import annotations

import json
import uuid

from smartlawai.adapters.base import AuditEvent, BackendInterface
from smartlawai.config import RERANK_FLOOR
from smartlawai.core.bm25 import BM25Retriever
from smartlawai.core.faithfulness import FaithfulnessEvaluator
from smartlawai.core.inlegalbert import InLegalBERTEncoder
from smartlawai.core.mistral_client import MODEL, complete
from smartlawai.core.rerank import Reranker

RRF_K = 60
DISCLAIMER = ("\n\n---\n*This is AI-generated legal information, not legal advice. "
              "Consult an advocate enrolled under the Advocates Act, 1961.*")

_ANSWER_SYS = ("You are an Indian legal assistant. Answer ONLY from the provided "
               "context passages. Cite passage numbers like [1]. If the context "
               "does not contain the answer, say you cannot answer from the document.")
_HYDE = "Write a short hypothetical legal answer to: {q}"


def _rrf(ranked_lists: list[list[str]]) -> dict[str, float]:
    fused: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            if cid:
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return fused


class AdvisoryRAG:
    def __init__(self, backend: BackendInterface, index_key: str = "indo_v1") -> None:
        self.be = backend
        self.enc = InLegalBERTEncoder()
        self.reranker = Reranker()
        self.faith = FaithfulnessEvaluator()
        self.index = backend.load_faiss_index(index_key)
        self.idmap = json.loads(backend._read_index_bytes(f"{index_key}.idmap").decode())
        chunks = backend.fetch_all_chunks()
        self.bm25 = BM25Retriever([c.chunk_id for c in chunks],
                                  [c.chunk_text for c in chunks])

    def _dense(self, query: str, k: int = 20) -> list[str]:
        qv = self.enc.embed([query]).astype("float32")
        _, ids = self.index.search(qv, k)
        return [self.idmap.get(str(int(i))) for i in ids[0] if int(i) >= 0]

    def answer(self, question: str, session_id: str = "anon") -> dict:
        hyde = complete(_HYDE.format(q=question), max_tokens=200, temperature=0.3)
        dense_ids = self._dense(question + " " + hyde, k=20)
        sparse_ids = [cid for cid, _ in self.bm25.search(question, k=20)]

        fused = _rrf([dense_ids, sparse_ids])
        cand_ids = [c for c, _ in sorted(fused.items(), key=lambda x: x[1],
                                         reverse=True)[:20] if c]
        cand = self.be.fetch_chunks_by_ids(cand_ids)
        by_id = {c.chunk_id: c.chunk_text for c in cand}

        top = self.reranker.rerank(
            question, [(c, by_id[c]) for c in cand_ids if c in by_id], 5)
        if not top or top[0][1] < RERANK_FLOOR:
            return self._refuse(question, session_id, "out_of_scope")

        ctx = "\n\n".join(f"[{i+1}] {by_id[cid]}" for i, (cid, _) in enumerate(top))
        ans = complete(f"Question: {question}\n\nContext:\n{ctx}",
                       system=_ANSWER_SYS, max_tokens=500)

        fa = self.faith.evaluate(ctx, ans)
        final = (ans + DISCLAIMER) if fa["passed"] else (
            "I cannot answer this reliably from the document (low faithfulness). "
            "Please consult a qualified advocate." + DISCLAIMER)

        self._audit(session_id, question, final, fa["score"])
        return {"answer": final, "faithfulness": fa, "passed": fa["passed"],
                "citations": [cid for cid, _ in top], "model": MODEL}

    def _refuse(self, q, session_id, reason) -> dict:
        msg = ("Your question appears outside the scope of the uploaded document. "
               "I can only answer from its contents." + DISCLAIMER)
        self._audit(session_id, q, msg, 0.0)
        return {"answer": msg, "faithfulness": {"score": 0.0, "passed": False,
                "reason": reason}, "passed": False, "citations": []}

    def _audit(self, session_id, q, a, score) -> None:
        self.be.store_audit(AuditEvent(
            event_id=f"ev-{uuid.uuid4().hex[:10]}", session_id=session_id,
            event_type="ADVISORY", endpoint="answer", query_text=q,
            response_text=a, faithfulness_score=score, pii_redacted=False))
