"""T1 (RESEARCH.md 8, PLAN.md pitfall checklist): hold out judgments
post-dating each model's training cutoff; report pre- and post-cutoff numbers
separately. Degrades gracefully -- skip-and-state-why -- matching the
convention already established for M1's own leakage checks. Never fabricates
a date (I2-style discipline applied to dates, not just verifier scores).

No pulled corpus (data/raw/summ, and IL-TUR's cjpe config once pulled) carries
a structured date field -- confirmed directly against both schemas. Dates are
extracted from the judgment text itself using the spaCy/regex tooling already
in this repo (core/ner.py) -- open-source-only, no new dependency, no new
data pull.

Design note on what was deliberately NOT built: an earlier draft of this
module also fell back to grabbing a year from the first case citation found
anywhere in the judgment text (core.ner.extract_entities()'s case_citations).
That was rejected: case citations inside a judgment are overwhelmingly
citations *to precedent*, not to the judgment itself, and a precedent must
predate the judgment that cites it. Using that year would systematically bias
resolved dates *earlier* than the true decision date -- which is actively
harmful here, since it would misclassify some genuinely post-cutoff judgments
as pre-cutoff, hiding exactly the contamination risk T1 exists to catch.
Returning 'unavailable' is safer than a biased guess in the wrong direction."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .config import MODEL_CUTOFF_DATES

_YEAR = re.compile(r"\b(18|19|20)\d{2}\b")

# Looks for the judgment's own stated decision date near the start of the
# text (headers/openers), e.g. "delivered on 15th January, 2023",
# "dated 01.01.2024", "decided on ...". Scoped to the first _HEADER_SCAN_CHARS
# characters so a "dated" appearing deep in a quoted precedent isn't mistaken
# for the judgment's own date.
_HEADER_PHRASE = re.compile(
    r"(?:delivered\s+on|pronounced\s+on|decided\s+on|dated)\s*[:\-]?\s*([^\n.]{0,40})",
    re.IGNORECASE)
_HEADER_SCAN_CHARS = 1000


@dataclass
class DateResolution:
    doc_id: str
    decision_date: date | None
    source: str  # "header_heuristic" | "unavailable"


def resolve_judgment_date(doc_id: str, text: str) -> DateResolution:
    """Tries the header-phrase heuristic only (see module docstring for why a
    citation-year fallback was rejected). Falls through to 'unavailable'
    rather than guessing -- an unreliable or systematically-biased date here
    corrupts the contamination split silently, which is worse than an honest
    gap."""
    header = text[:_HEADER_SCAN_CHARS]
    m = _HEADER_PHRASE.search(header)
    if m:
        year_m = _YEAR.search(m.group(1))
        if year_m:
            return DateResolution(doc_id, date(int(year_m.group()), 1, 1),
                                  "header_heuristic")
    return DateResolution(doc_id, None, "unavailable")


def split_by_cutoff(resolutions: list[DateResolution], model_id: str) -> dict:
    """{'pre_cutoff': [...], 'post_cutoff': [...], 'undated_skipped': [...]}.
    Raises KeyError if model_id isn't in MODEL_CUTOFF_DATES -- a config gap to
    fill deliberately (baselines/config.py), never silently skipped past."""
    if model_id not in MODEL_CUTOFF_DATES:
        raise KeyError(
            f"No verified cutoff date for {model_id!r} in baselines/config.py's "
            f"MODEL_CUTOFF_DATES -- add one with a source citation before running "
            f"this split.")
    cutoff = MODEL_CUTOFF_DATES[model_id]
    pre: list[DateResolution] = []
    post: list[DateResolution] = []
    undated: list[DateResolution] = []
    for r in resolutions:
        if r.decision_date is None:
            undated.append(r)
        elif r.decision_date < cutoff:
            pre.append(r)
        else:
            post.append(r)
    return {"pre_cutoff": pre, "post_cutoff": post, "undated_skipped": undated}
