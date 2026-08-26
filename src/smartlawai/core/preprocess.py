"""Normalisation for Indian legal text: Unicode NFC, Indic normalisation,
whitespace cleanup, code-switch detection. Flips preprocess_status -> DONE."""
from __future__ import annotations

import re
import unicodedata

from smartlawai.adapters.base import BackendInterface

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")

try:  # optional: indic-nlp-library
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    _HINDI_NORM = IndicNormalizerFactory().get_normalizer("hi")
except Exception:
    _HINDI_NORM = None


def is_code_switched(text: str) -> bool:
    return bool(_DEVANAGARI.search(text)) and bool(re.search(r"[A-Za-z]", text))


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    if _HINDI_NORM and _DEVANAGARI.search(text):
        text = _HINDI_NORM.normalize(text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()


def preprocess_doc(doc_id: str, raw_text: str, backend: BackendInterface) -> str:
    cleaned = normalize(raw_text)
    backend.mark_preprocessed(doc_id, "DONE")
    return cleaned
