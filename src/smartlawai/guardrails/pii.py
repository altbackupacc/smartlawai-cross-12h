"""PII detection and redaction (DPDPA 2023 compliant).

Implements PLAN.md M5.7:
- Correct ordering: extract entities -> store spans separately -> redact -> store.
- Allowed entity set: PHONE, EMAIL, IN_AADHAAR, IN_PAN, CREDIT_CARD only.
- NEVER redact PERSON, LOCATION, or DATE_TIME (they are the substance of a judgment).
- Fail-closed architecture (never silently fail open to unredacted text on error).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Strict entity set per PLAN.md M5.7
# Explicitly NEVER includes PERSON, LOCATION, DATE_TIME
ALLOWED_PII_ENTITIES = [
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "IN_PAN",
    "IN_AADHAAR",
]

# Regex fallbacks for Indian PII when Presidio analyzer is unavailable
_PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b")
_AADHAAR_REGEX = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b")
_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_PHONE_REGEX = re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b")


@dataclass
class PIISpan:
    """A detected PII entity span."""

    entity_type: str
    start: int
    end: int
    value: str


@dataclass
class PIIRedactionResult:
    """Result of PII extraction and redaction."""

    redacted_text: str
    spans: list[PIISpan] = field(default_factory=list)
    redacted_count: int = 0
    status: str = "ok"  # "ok" | "failed_closed"
    reason: str = ""


_analyzer = None
_anonymizer = None


def _get_presidio_engines() -> tuple[Any, Any] | None:
    """Lazily load Presidio engines."""
    global _analyzer, _anonymizer
    if _analyzer is not None and _anonymizer is not None:
        return _analyzer, _anonymizer

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
        return _analyzer, _anonymizer
    except Exception as e:  # noqa: BLE001
        logger.debug("Presidio not available: %s, falling back to regex", e)
        return None


def extract_pii_spans(text: str, lang: str = "en") -> list[PIISpan]:
    """Extract PII spans without altering text (Step 1 in M5.7 order)."""
    if not text:
        return []

    spans: list[PIISpan] = []
    engines = _get_presidio_engines()
    if engines is not None:
        try:
            analyzer, _ = engines
            results = analyzer.analyze(
                text=text,
                entities=ALLOWED_PII_ENTITIES,
                language=lang,
            )
            for r in results:
                spans.append(
                    PIISpan(
                        entity_type=r.entity_type,
                        start=r.start,
                        end=r.end,
                        value=text[r.start : r.end],
                    )
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Presidio extraction failed: %s; falling back to regex", e)

    # Always ensure Indian statutory PII (IN_PAN and IN_AADHAAR) and core patterns are covered
    for m in _PAN_REGEX.finditer(text):
        if not any(s.start <= m.start() and s.end >= m.end() for s in spans):
            spans.append(PIISpan(entity_type="IN_PAN", start=m.start(), end=m.end(), value=m.group(0)))
    for m in _AADHAAR_REGEX.finditer(text):
        if not any(s.start <= m.start() and s.end >= m.end() for s in spans):
            spans.append(PIISpan(entity_type="IN_AADHAAR", start=m.start(), end=m.end(), value=m.group(0)))
    for m in _EMAIL_REGEX.finditer(text):
        if not any(s.start <= m.start() and s.end >= m.end() for s in spans):
            spans.append(PIISpan(entity_type="EMAIL_ADDRESS", start=m.start(), end=m.end(), value=m.group(0)))
    for m in _PHONE_REGEX.finditer(text):
        if not any(s.start <= m.start() and s.end >= m.end() for s in spans):
            spans.append(PIISpan(entity_type="PHONE_NUMBER", start=m.start(), end=m.end(), value=m.group(0)))

    # Sort spans by start index
    spans.sort(key=lambda s: s.start)
    return spans


def redact_pii(text: str, lang: str = "en") -> PIIRedactionResult:
    """Extract entities -> store spans separately -> redact -> return result.

    Fails closed: if an unexpected exception occurs, returns status="failed_closed".
    """
    if not text:
        return PIIRedactionResult(redacted_text="", spans=[], redacted_count=0)

    try:
        spans = extract_pii_spans(text, lang=lang)

        if not spans:
            return PIIRedactionResult(redacted_text=text, spans=[], redacted_count=0)

        # Apply redactions from end of string to beginning to preserve character offsets
        redacted = list(text)
        sorted_spans = sorted(spans, key=lambda s: s.start, reverse=True)

        for s in sorted_spans:
            placeholder = f"<{s.entity_type}>"
            redacted[s.start : s.end] = list(placeholder)

        result_text = "".join(redacted)
        return PIIRedactionResult(
            redacted_text=result_text,
            spans=spans,
            redacted_count=len(spans),
            status="ok",
        )
    except Exception as e:  # noqa: BLE001
        logger.error("PII redaction encountered an error; failing closed: %s", e)
        return PIIRedactionResult(
            redacted_text="",
            spans=[],
            redacted_count=0,
            status="failed_closed",
            reason=str(e),
        )


def redact(text: str, lang: str = "en") -> tuple[str, int]:
    """Backward-compatible tuple interface."""
    res = redact_pii(text, lang=lang)
    if res.status != "ok":
        # Fail closed
        return "", 0
    return res.redacted_text, res.redacted_count
