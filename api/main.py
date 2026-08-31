"""SmartLawAI FastAPI. Endpoints: session, upload, ocr, summarise, clauses,
entities (NER), ask, health.

Changes from v0.1:
- POST /entities/{doc_id}  — Indian legal NER (hybrid regex + spaCy)
- Audit logging wired into every mutating endpoint via _audit()
- BCI disclaimer attached to /ask and /summarise responses
- PII redaction applied to /summarise, /clauses, /ask (not just /ocr)
- Error handling with proper HTTP 404/500 responses
- /summarise and /clauses changed from GET to POST (side-effecting ops)
- Temp files cleaned up after upload
"""
from __future__ import annotations

import logging
import os
import tempfile
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from smartlawai.adapters.base import AuditEvent
from smartlawai.adapters.factory import get_backend
from smartlawai.guardrails.disclaimer import BCI_DISCLAIMER
from smartlawai.guardrails.disclaimer import attach as attach_disclaimer
from smartlawai.guardrails.pii import redact
from smartlawai.pipeline import Pipeline
from smartlawai.scope import Scope

log = logging.getLogger("smartlawai.api")

app = FastAPI(title="SmartLawAI", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
be = get_backend()
_extractor = None
_pipeline = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class ScopeBody(BaseModel):
    doc_ids: list[str]
    owner_id: str = "anon"


class AskBody(BaseModel):
    question: str
    scope: ScopeBody  # required -- I1 enforced at the HTTP boundary too
    session_id: str = "anon"  # unrelated to retrieval scope; used only for audit logging


def _audit(session_id: str, event_type: str, endpoint: str,
           query_text: str = "", response_text: str = "",
           faithfulness_score: float | None = None, pii_redacted: bool = False):
    """Centralised audit logging for all endpoints."""
    try:
        be.store_audit(AuditEvent(
            event_id=f"ev-{uuid.uuid4().hex[:10]}", session_id=session_id,
            event_type=event_type, endpoint=endpoint,
            query_text=query_text[:500], response_text=response_text[:500],
            faithfulness_score=faithfulness_score, pii_redacted=pii_redacted))
    except Exception:
        log.exception("Audit logging failed (non-fatal)")


def _require_doc(doc_id: str) -> str:
    """Fetch document text or raise 404."""
    text = be.fetch_document_text(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return text


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.post("/session")
def create_session():
    return {"session_id": f"s-{uuid.uuid4().hex[:12]}"}


@app.post("/upload")
async def upload(file: UploadFile = File(...), doc_type: str = "JUDGMENT",
                 session_id: str = "anon", owner_id: str = "anon"):
    global _pipeline
    suffix = os.path.splitext(file.filename or "")[1]
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            path = tmp.name
        _pipeline = _pipeline or Pipeline(be)
        doc_id, trace = _pipeline.ingest(path, doc_type=doc_type, source="upload",
                                         owner_id=owner_id)
        ingest_stage = next((s for s in trace.stages if s.name == "ingest"), None)
        detail = ingest_stage.detail if ingest_stage else {}
        _audit(session_id, "UPLOAD", "/upload",
               query_text=file.filename or "", response_text=doc_id)
        return {"doc_id": doc_id, "status": detail.get("ocr_status", "OCR_FAIL"),
                "language": detail.get("language", "unknown"),
                "pages": detail.get("pages", 0), "trace": trace.to_dict()}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


@app.get("/ocr/{doc_id}")
def ocr_text(doc_id: str):
    text = _require_doc(doc_id)
    redacted, n = redact(text)
    _audit("anon", "OCR", "/ocr", query_text=doc_id, pii_redacted=n > 0)
    return {"doc_id": doc_id, "text": redacted, "pii_redacted_count": n}


@app.post("/summarise/{doc_id}")
def summarise(doc_id: str, session_id: str = "anon"):
    try:
        text = _require_doc(doc_id)
        from smartlawai.core.summarize import summarise_document
        summaries = summarise_document(doc_id, text)
        be.store_summaries(summaries)
        final = next((s for s in summaries if s.level == "FINAL"), summaries[-1])
        summary_text = final.summary_text
        # PII redaction on output
        summary_text, n_pii = redact(summary_text)
        # Attach BCI disclaimer
        summary_text = attach_disclaimer(summary_text)
        _audit(session_id, "SUMMARISE", "/summarise",
               query_text=doc_id, response_text=summary_text[:300], pii_redacted=n_pii > 0)
        return {"doc_id": doc_id, "summary": summary_text,
                "disclaimer": BCI_DISCLAIMER}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Summarise failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/clauses/{doc_id}")
def clauses(doc_id: str, session_id: str = "anon"):
    global _extractor
    try:
        text = _require_doc(doc_id)
        from smartlawai.core.clauses import ClauseExtractor
        _extractor = _extractor or ClauseExtractor()
        found = _extractor.extract(doc_id, text)
        be.store_clauses(found)
        # PII redact clause texts
        clause_dicts = []
        for c in found:
            ct, _ = redact(c.clause_text)
            clause_dicts.append({
                "type": c.clause_type, "text": ct, "risk": c.risk_tier,
                "score": c.risk_score, "statute": c.statute_ref})
        _audit(session_id, "CLAUSES", "/clauses",
               query_text=doc_id, response_text=f"{len(found)} clauses extracted")
        return {"doc_id": doc_id, "clauses": clause_dicts}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Clause extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/entities/{doc_id}")
def entities(doc_id: str, session_id: str = "anon"):
    """Extract Indian legal named entities (NER) from a document."""
    try:
        text = _require_doc(doc_id)
        from smartlawai.core.ner import extract_entities
        result = extract_entities(doc_id, text)
        _audit(session_id, "NER", "/entities",
               query_text=doc_id,
               response_text=f"{len(result.entities)} entities extracted")
        return {
            "doc_id": doc_id,
            "total_entities": len(result.entities),
            "statutes": result.statutes,
            "case_citations": result.case_citations,
            "courts": result.courts,
            "judges": result.judges,
            "parties": result.parties,
            "dates": result.dates,
            "monetary_values": result.monetary_values,
            "legal_provisions": result.legal_provisions,
            "organizations": result.organizations,
            "locations": result.locations,
            "entities": [
                {"text": e.text, "label": e.label, "start": e.start,
                 "end": e.end, "source": e.source}
                for e in result.entities
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("NER extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask(body: AskBody):
    global _pipeline
    try:
        scope = Scope(doc_ids=tuple(body.scope.doc_ids), owner_id=body.scope.owner_id)
        _pipeline = _pipeline or Pipeline(be)
        result = _pipeline.ask(body.question, scope)
        answer_text, n_pii = redact(result.answer)
        _audit(body.session_id, "ASK", "/ask", query_text=body.question,
               response_text=answer_text, pii_redacted=n_pii > 0)
        return {"answer": answer_text, "decision": result.decision,
                "trace": result.trace.to_dict(), "disclaimer": BCI_DISCLAIMER}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "backend": os.environ.get("SMARTLAW_BACKEND", "local")}
