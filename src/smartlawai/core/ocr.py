"""OCR + text extraction for Indian legal docs (PDF/DOCX/image/txt).
Writes an INGESTION_LOG row per file and a document row on success.
Heavy deps (pytesseract, pdf2image) imported lazily so the package imports
without them; only extraction calls require them installed."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from smartlawai.adapters.base import BackendInterface, Document, IngestionRecord

OCR_ENGINE = "tesseract-5"
DIGITAL_TEXT_MIN_CHARS = 100  # below this a PDF is treated as scanned -> OCR


@dataclass
class IngestResult:
    doc_id: str
    ocr_status: str          # DIGITAL | OCR_OK | OCR_FAIL
    lang_detected: str
    num_pages: int
    text: str
    error: str | None = None


def _extract_pdf_text(path: str) -> tuple[str, int]:
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages), len(pages)


def _ocr_pdf(path: str, lang: str = "eng+hin") -> tuple[str, int]:
    import pytesseract
    from pdf2image import convert_from_path
    images = convert_from_path(path, dpi=300)
    text = "\n".join(pytesseract.image_to_string(img, lang=lang) for img in images)
    return text, len(images)


def _detect_lang(text: str) -> str:
    if not text.strip():
        return "unknown"
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "unknown"


def extract_text(path: str) -> IngestResult:
    doc_id = f"doc-{uuid.uuid4().hex[:10]}"
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            text, n = _extract_pdf_text(path)
            if len(text.strip()) >= DIGITAL_TEXT_MIN_CHARS:
                status = "DIGITAL"
            else:
                text, n = _ocr_pdf(path)
                status = "OCR_OK"
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif"):
            import pytesseract
            text = pytesseract.image_to_string(path, lang="eng+hin")
            n, status = 1, "OCR_OK"
        elif ext in (".txt", ".md"):
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            n, status = 1, "DIGITAL"
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return IngestResult(doc_id, status, _detect_lang(text), n, text)
    except Exception as e:  # noqa: BLE001 - log failure, never crash the batch
        return IngestResult(doc_id, "OCR_FAIL", "unknown", 0, "", error=str(e))


def ingest_file(
    path: str,
    backend: BackendInterface,
    doc_type: str,
    source: str,
    session_id: str | None = None,
) -> IngestResult:
    res = extract_text(path)
    backend.store_ingestion(IngestionRecord(
        ingest_id=f"ing-{uuid.uuid4().hex[:10]}", doc_id=res.doc_id,
        stage_path=path, ocr_engine=OCR_ENGINE, ocr_status=res.ocr_status,
        lang_detected=res.lang_detected, preprocess_status="PENDING",
        error_msg=res.error))
    if res.ocr_status != "OCR_FAIL":
        backend.upload_doc(path, Document(
            doc_id=res.doc_id, filename=os.path.basename(path), doc_type=doc_type,
            language=res.lang_detected, source=source, storage_uri="",
            num_pages=res.num_pages, ocr_applied=(res.ocr_status == "OCR_OK"),
            raw_text=res.text, session_id=session_id))
    return res
