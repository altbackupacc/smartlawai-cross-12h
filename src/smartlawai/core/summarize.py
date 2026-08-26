"""Hierarchical summarisation: segment -> summarise each section -> synthesise.
Avoids context-window limits on long judgments. Uses the Mistral HTTP client."""
from __future__ import annotations

import uuid

from smartlawai.adapters.base import Summary
from smartlawai.core.mistral_client import MODEL, complete

_SYS = ("You are a legal assistant. Summarise in plain language for a layperson. "
        "Use ONLY information present in the text. Do not invent statutes, "
        "citations, dates, or holdings.")
_SECTION_PROMPT = "Summarise this section of an Indian legal document:\n\n{chunk}"
_SYNTH_PROMPT = ("Combine these section summaries into one coherent plain-language "
                 "summary (max 250 words) covering facts, issue, holding, "
                 "reasoning:\n\n{parts}")


def _segment(text: str, max_chars: int = 3000) -> list[str]:
    paras = text.split("\n\n")
    buf, out = "", []
    for p in paras:
        if len(buf) + len(p) > max_chars:
            if buf.strip():
                out.append(buf)
            buf = p
        else:
            buf += "\n\n" + p
    if buf.strip():
        out.append(buf)
    return out or [text[:max_chars]]


def summarise_document(doc_id: str, text: str) -> list[Summary]:
    sections = _segment(text)
    summaries: list[Summary] = []
    section_texts: list[str] = []

    for i, sec in enumerate(sections):
        s = complete(_SECTION_PROMPT.format(chunk=sec), system=_SYS, max_tokens=256)
        section_texts.append(s)
        summaries.append(Summary(
            summary_id=f"sum-{uuid.uuid4().hex[:10]}", doc_id=doc_id,
            section_label=f"section_{i}", level="SECTION", summary_text=s, model=MODEL))

    final = complete(_SYNTH_PROMPT.format(parts="\n\n".join(section_texts)),
                     system=_SYS, max_tokens=400)
    summaries.append(Summary(
        summary_id=f"sum-{uuid.uuid4().hex[:10]}", doc_id=doc_id,
        section_label="FINAL", level="FINAL", summary_text=final, model=MODEL))
    return summaries
