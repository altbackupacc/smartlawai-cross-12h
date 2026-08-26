"""PII redaction via Microsoft Presidio (DPDPA 2023). Lazy import; degrades
gracefully (returns text unchanged) if presidio is not installed."""
from __future__ import annotations

_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD",
             "IN_PAN", "IN_AADHAAR", "LOCATION", "DATE_TIME"]

_analyzer = None
_anon = None


def _engines():
    global _analyzer, _anon
    if _analyzer is None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        _analyzer = AnalyzerEngine()
        _anon = AnonymizerEngine()
    return _analyzer, _anon


def redact(text: str, lang: str = "en") -> tuple[str, int]:
    try:
        analyzer, anon = _engines()
        results = analyzer.analyze(text=text, entities=_ENTITIES, language=lang)
        out = anon.anonymize(text=text, analyzer_results=results)
        return out.text, len(results)
    except Exception:
        return text, 0
