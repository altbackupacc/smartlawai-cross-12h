"""Extractiveness: how much of a generated text is copied verbatim from source.

RESEARCH.md §5 makes this mandatory alongside faithfulness, for a specific
reason worth restating at the top of the module that computes it:

    HHEM rewards entailment. A model that copies source sentences verbatim
    scores near-perfect faithfulness and is a useless summariser. Reporting
    faithfulness without extractiveness invites the reviewer question "did you
    just train a copier?" and we must have the answer.

So a faithfulness number is only ever reported next to one of these. High
extractiveness plus high faithfulness is the honest reviewer-facing answer;
high faithfulness alone is not an answer at all.

Imports nothing from `smartlawai` (OPS.md §8).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from eval import config

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    """Lowercased word tokens. Punctuation-insensitive on purpose: copying a
    sentence and changing its comma is still copying."""
    return _TOKEN.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def extractiveness(pred: str, source: str,
                   n: int = config.EXTRACTIVENESS_NGRAM) -> float | None:
    """Fraction of pred's n-grams (default 4-gram) that appear verbatim in source.

    1.0 means every n-gram of the output was lifted from the source; 0.0 means
    none was.

    Returns None -- not 0.0 -- when the prediction is shorter than n tokens, so
    there are no n-grams to judge. A one-word answer has not been shown to be
    original; it has not been measured. (I2 applied to a metric's own input.)
    """
    if n <= 0:
        raise ValueError("n-gram size must be positive")
    pred_tokens, source_tokens = _tokens(pred), _tokens(source)
    pred_ngrams = _ngrams(pred_tokens, n)
    if not pred_ngrams:
        return None
    source_ngrams = set(_ngrams(source_tokens, n))
    if not source_ngrams:
        return 0.0
    hits = sum(1 for g in pred_ngrams if g in source_ngrams)
    return round(hits / len(pred_ngrams), 4)


def coverage(pred: str, source: str) -> float | None:
    """Fraction of pred's TOKENS that appear anywhere in source (unigram overlap).

    Reported next to extractiveness because they separate two different copying
    behaviours: high coverage with low extractiveness is genuine paraphrase using
    the source's vocabulary, which is what a good legal summary looks like.
    """
    pred_tokens = _tokens(pred)
    if not pred_tokens:
        return None
    source_vocab = set(_tokens(source))
    return round(sum(1 for t in pred_tokens if t in source_vocab) / len(pred_tokens), 4)


def longest_copied_span(pred: str, source: str) -> int:
    """Length in tokens of the longest verbatim run shared with the source.

    A single long copied passage and many short shared phrases can produce the
    same 4-gram extractiveness; this tells them apart, which is what a reviewer
    asking "did you just quote the headnote?" actually wants to know.
    """
    pred_tokens, source_tokens = _tokens(pred), _tokens(source)
    if not pred_tokens or not source_tokens:
        return 0
    # Rolling DP over the suffix lengths -- O(len(pred) * len(source)) time,
    # O(len(source)) memory. Inputs here are single passages, not corpora.
    previous = [0] * (len(source_tokens) + 1)
    best = 0
    for token in pred_tokens:
        current = [0] * (len(source_tokens) + 1)
        for j, src_token in enumerate(source_tokens, start=1):
            if token == src_token:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


@dataclass
class ExtractivenessReport:
    """The three numbers reported together, plus the n they were computed at."""

    ngram: int
    extractiveness: float | None
    coverage: float | None
    longest_copied_span: int
    n_pred_tokens: int
    status: str = "ok"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ngram": self.ngram, "extractiveness": self.extractiveness,
                "coverage": self.coverage,
                "longest_copied_span": self.longest_copied_span,
                "n_pred_tokens": self.n_pred_tokens,
                "status": self.status, "reason": self.reason}


def extractiveness_report(pred: str, source: str,
                          n: int = config.EXTRACTIVENESS_NGRAM,
                          ) -> ExtractivenessReport:
    """All three copying measures for one (prediction, source) pair."""
    pred_tokens = _tokens(pred)
    value = extractiveness(pred, source, n)
    return ExtractivenessReport(
        ngram=n,
        extractiveness=value,
        coverage=coverage(pred, source),
        longest_copied_span=longest_copied_span(pred, source),
        n_pred_tokens=len(pred_tokens),
        status="ok" if value is not None else "unavailable",
        reason="" if value is not None else f"prediction_shorter_than_{n}_tokens")


def mean_extractiveness(preds: list[str], sources: list[str],
                        n: int = config.EXTRACTIVENESS_NGRAM) -> float | None:
    """Corpus-level mean over pairs long enough to measure.

    Pairs too short to have an n-gram are skipped, not scored 0.0 -- otherwise a
    system that refuses often would look maximally original.
    """
    if len(preds) != len(sources):
        raise ValueError(f"preds and sources must align: "
                         f"{len(preds)} vs {len(sources)}")
    values = [v for v in (extractiveness(p, s, n) for p, s in zip(preds, sources))
              if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)
