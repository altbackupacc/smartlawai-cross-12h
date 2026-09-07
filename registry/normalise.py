"""Normalise raw citation text to canonical registry identifiers.

Input is whatever `smartlawai.core.ner.extract_entities()` emits as a `STATUTE`
or `PROVISION` entity. Output is a canonical act id plus a parsed section
reference. Shared by the corpus frequency analysis and by `resolver.py`.

Deliberately conservative: anything not confidently recognised returns None so
the caller can count it as unresolved. CLAUDE.md I2 -- never guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Canonical acts
# --------------------------------------------------------------------------- #
# id -> (short_title, year, aliases). Aliases are matched case-insensitively,
# longest-first, so "Indian Penal Code" wins over a bare "IPC" substring.
ACTS: dict[str, tuple[str, int, tuple[str, ...]]] = {
    # --- the criminal six: the natural experiment (RESEARCH.md 2) ---
    "ipc-1860":  ("Indian Penal Code", 1860,
                  ("indian penal code", "i.p.c.", "ipc", "penal code")),
    "crpc-1973": ("Code of Criminal Procedure", 1973,
                  ("code of criminal procedure", "criminal procedure code",
                   "cr.p.c.", "crpc", "cr p c")),
    "iea-1872":  ("Indian Evidence Act", 1872,
                  ("indian evidence act", "evidence act", "i.e.a.")),
    "bns-2023":  ("Bharatiya Nyaya Sanhita", 2023,
                  ("bharatiya nyaya sanhita", "bns")),
    "bnss-2023": ("Bharatiya Nagarik Suraksha Sanhita", 2023,
                  ("bharatiya nagarik suraksha sanhita", "bnss")),
    "bsa-2023":  ("Bharatiya Sakshya Adhiniyam", 2023,
                  ("bharatiya sakshya adhiniyam", "bsa")),
    # --- remaining seed statutes (PLAN.md M4) ---
    "contract-1872":  ("Indian Contract Act", 1872, ("indian contract act", "contract act")),
    "arbitration-1996": ("Arbitration and Conciliation Act", 1996,
                         ("arbitration and conciliation act", "arbitration act")),
    "consumer-2019":  ("Consumer Protection Act", 2019, ("consumer protection act",)),
    "cpc-1908":       ("Code of Civil Procedure", 1908,
                       ("code of civil procedure", "civil procedure code", "c.p.c.", "cpc")),
    "ni-1881":        ("Negotiable Instruments Act", 1881, ("negotiable instruments act", "ni act")),
    "it-2000":        ("Information Technology Act", 2000,
                       ("information technology act", "it act")),
    "stamp-1899":     ("Indian Stamp Act", 1899, ("indian stamp act", "stamp act")),
    # --- not a seed statute, but ner.py emits it and it is never repealed;
    #     carried so Article citations are classified rather than dropped ---
    "constitution-1950": ("Constitution of India", 1950,
                          ("constitution of india", "constitution")),
}

# Longest alias first so "indian contract act" beats "contract act", and
# "code of criminal procedure" beats "cpc".
_ALIAS_INDEX: list[tuple[str, str]] = sorted(
    ((alias, act_id) for act_id, (_, _, aliases) in ACTS.items() for alias in aliases),
    key=lambda kv: -len(kv[0]),
)

# Provision keywords that _SECTION_REF can emit.
_PROVISION_KINDS = {
    "section": "s", "sec": "s", "s": "s",
    "article": "art", "art": "art",
    "rule": "r", "order": "o", "clause": "cl",
}

_KIND_RE = re.compile(
    r"^\s*(section|sec|s|article|art|rule|order|clause)\b\.?\s*", re.IGNORECASE
)
# 302 | 302A | 302-304 | 302 to 304, plus trailing (a)(ii) sub-clauses
_NUM_RE = re.compile(
    r"(?P<num>\d+[A-Za-z]?)"
    r"(?:\s*(?:to|-|–)\s*(?P<num_to>\d+[A-Za-z]?))?"
    r"(?P<subs>(?:\s*\([a-zA-Z0-9]+\))*)"
)
_SUB_RE = re.compile(r"\(([a-zA-Z0-9]+)\)")
# With no keyword present, a reference is only valid if it is bare digits
# ("302", "302A", "302-304"). This is what stops "Indian Penal Code, 1860"
# from parsing as section 1860.
_BARE_NUM_RE = re.compile(
    r"^\d+[A-Za-z]?(?:\s*(?:to|-|–)\s*\d+[A-Za-z]?)?(?:\s*\([a-zA-Z0-9]+\))*\s*$"
)


@dataclass(frozen=True)
class SectionRef:
    """A parsed provision reference, act-agnostic."""

    kind: str                  # "s" | "art" | "r" | "o" | "cl"
    number: str                # "302", "302A" -- always the FIRST number of a range
    number_to: str | None = None   # set when the citation was a range
    subclauses: tuple[str, ...] = ()

    @property
    def is_range(self) -> bool:
        return self.number_to is not None


# Short forms used when rendering a note back to a human, e.g.
# "repealed 2024-07-01; see BNS 318". Derived forms would give "CRPC-1973".
ABBREV: dict[str, str] = {
    "ipc-1860": "IPC", "crpc-1973": "CrPC", "iea-1872": "IEA",
    "bns-2023": "BNS", "bnss-2023": "BNSS", "bsa-2023": "BSA",
    "contract-1872": "Contract Act", "arbitration-1996": "Arbitration Act",
    "consumer-2019": "Consumer Protection Act", "cpc-1908": "CPC",
    "ni-1881": "NI Act", "it-2000": "IT Act", "stamp-1899": "Stamp Act",
    "constitution-1950": "Constitution",
}


def abbrev(act_id: str) -> str:
    return ABBREV.get(act_id, act_id)


def normalise_act(text: str) -> str | None:
    """Map any statute mention to a canonical act id, or None if unrecognised."""
    if not text:
        return None
    low = " " + re.sub(r"\s+", " ", text.lower().strip()) + " "
    # Strip a leading provision reference: "Section 420 IPC" -> " ipc "
    low = _KIND_RE.sub(" ", low)
    for alias, act_id in _ALIAS_INDEX:
        # Word-boundary match so "bns" does not fire inside "bnss".
        if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", low):
            return act_id
    return None


def parse_section_ref(text: str) -> SectionRef | None:
    """Parse 'Section 302A(1)(b)' / 'Art. 21' / 'Section 302 to 304'.

    Requires an explicit provision keyword ("Section", "Art.", "Rule", ...) or a
    bare number. Without that guard a plain act mention carrying its year --
    "Indian Penal Code, 1860" -- parses as section 1860 and fabricates a section
    that does not exist (`ipc-1860-s1860`). Returning None here is what keeps
    those out of the registry and out of the frequency counts.
    """
    if not text:
        return None
    t = re.sub(r"\s+", " ", text.strip())
    m_kind = _KIND_RE.match(t)
    kind = "s"
    if m_kind:
        kind = _PROVISION_KINDS.get(m_kind.group(1).lower().rstrip("."), "s")
        t = t[m_kind.end():]
    elif not _BARE_NUM_RE.match(t):
        return None
    m = _NUM_RE.search(t)
    if not m:
        return None
    subs = tuple(_SUB_RE.findall(m.group("subs") or ""))
    return SectionRef(
        kind=kind,
        number=m.group("num").upper(),
        number_to=(m.group("num_to").upper() if m.group("num_to") else None),
        subclauses=subs,
    )


def is_compatible(act_id: str, kind: str) -> bool:
    """Can `act_id` contain a provision of this kind?

    Structural facts about the statutes, not heuristics. Without this check the
    nearest-preceding-act fallback happily produces `ipc-1860-art22` (the IPC
    has no Articles) and `constitution-1950-s4` (the Constitution has no
    Sections) -- both observed on this corpus before the guard existed.
    """
    if kind == "art":
        # Articles are a Constitution structure. No seed statute uses them.
        return act_id == "constitution-1950"
    if kind in ("s", "cl"):
        # Every act is divided into Sections; the Constitution is not.
        return act_id != "constitution-1950"
    if kind in ("o", "r"):
        # Orders and Rules are the CPC's schedule structure.
        return act_id == "cpc-1908"
    return True


def section_id(act_id: str, ref: SectionRef) -> str:
    """Canonical section identifier, e.g. 'ipc-1860-s420'.

    Sub-clauses are deliberately NOT part of the id: the registry tracks
    in-force status at section granularity, and a sub-clause inherits it.
    """
    return f"{act_id}-{ref.kind}{ref.number}"
