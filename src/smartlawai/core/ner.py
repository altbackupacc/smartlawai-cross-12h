"""Indian Legal Named Entity Recognition — hybrid regex + spaCy approach.

Regex extractors cover structured legal patterns:
  - IPC / CrPC / BNS / CPC / BNSS section references
  - Statute + section citations (e.g. "Indian Contract Act, 1872 (S.27)")
  - Case citations: AIR, SCC, SCR, and neutral citation formats
  - Monetary values (₹ / Rs. / INR)
  - Dates (multiple Indian date formats)
  - Case numbers (e.g. "W.P.(C) No. 1234/2023")

spaCy provides general-purpose NER for:
  - PERSON (judges, parties, advocates)
  - ORG (courts, organisations)
  - GPE / LOC (jurisdictions, locations)

Results are merged, deduplicated, normalised, and returned as a unified schema.
Lazy imports keep the module import-time lightweight.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Entity schema
# --------------------------------------------------------------------------- #
@dataclass
class Entity:
    """A single extracted entity."""
    text: str
    label: str       # e.g. STATUTE, CASE_CITATION, COURT, PERSON, DATE, MONEY, ...
    start: int = -1
    end: int = -1
    source: str = ""  # "regex" | "spacy"


@dataclass
class NERResult:
    """Aggregated extraction result for a document."""
    doc_id: str
    entities: list[Entity] = field(default_factory=list)
    statutes: list[str] = field(default_factory=list)
    case_citations: list[str] = field(default_factory=list)
    courts: list[str] = field(default_factory=list)
    judges: list[str] = field(default_factory=list)
    parties: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    monetary_values: list[str] = field(default_factory=list)
    legal_provisions: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Regex patterns — Indian legal domain
# --------------------------------------------------------------------------- #

# Statutes with section references
_STATUTE_SECTION = re.compile(
    r"(?:(?:Section|Sec\.|S\.)\s*\d+[A-Za-z]?(?:\s*(?:to|-)\s*\d+[A-Za-z]?)?(?:\s*\([a-zA-Z0-9]+\))*"
    r"\s*(?:of\s+(?:the\s+)?)?)"
    r"(?:Indian Penal Code|IPC|Bharatiya Nyaya Sanhita|BNS"
    r"|Code of Criminal Procedure|CrPC|Cr\.P\.C\.|Bharatiya Nagarik Suraksha Sanhita|BNSS"
    r"|Code of Civil Procedure|CPC|C\.P\.C\."
    r"|Indian Contract Act|Indian Evidence Act|Bharatiya Sakshya Adhiniyam"
    r"|Constitution of India|Indian Stamp Act"
    r"|Arbitration and Conciliation Act|Negotiable Instruments Act"
    r"|Companies Act|Information Technology Act|IT Act"
    r"|Consumer Protection Act|RERA"
    r"|Transfer of Property Act|Specific Relief Act"
    r"|Limitation Act|Registration Act"
    r"|Motor Vehicles Act|NDPS Act"
    r"|Prevention of Corruption Act|PMLA"
    r"|CGST Act|SGST Act|IGST Act"
    r"|Right to Information Act|RTI Act)"
    r"(?:[,\s]+\d{4})?",
    re.IGNORECASE,
)

# Bare statute reference (Act name with year)
_BARE_STATUTE = re.compile(
    r"\b(?:Indian Penal Code|IPC|Bharatiya Nyaya Sanhita|BNS"
    r"|Code of Criminal Procedure|CrPC|Cr\.P\.C\.|Bharatiya Nagarik Suraksha Sanhita|BNSS"
    r"|Code of Civil Procedure|CPC|C\.P\.C\."
    r"|Indian Contract Act|Indian Evidence Act|Bharatiya Sakshya Adhiniyam"
    r"|Constitution of India|Indian Stamp Act"
    r"|Arbitration and Conciliation Act|Negotiable Instruments Act"
    r"|Companies Act|Information Technology Act|IT Act"
    r"|Consumer Protection Act|RERA"
    r"|Transfer of Property Act|Specific Relief Act"
    r"|Limitation Act|Registration Act"
    r"|Motor Vehicles Act|NDPS Act"
    r"|Prevention of Corruption Act|PMLA"
    r"|CGST Act|SGST Act|IGST Act"
    r"|Right to Information Act|RTI Act)"
    r"(?:[,\s]+\d{4})?",
    re.IGNORECASE,
)

# Bare section references (e.g. "Section 302", "S. 420", "Art. 21")
_SECTION_REF = re.compile(
    r"\b(?:Section|Sec\.|S\.|Article|Art\.|Rule|Order|Clause)\s+"
    r"\d+[A-Za-z]?(?:\s*(?:to|-)\s*\d+[A-Za-z]?)?"
    r"(?:\s*\([a-zA-Z0-9]+\))*",
    re.IGNORECASE,
)

# Case citations — AIR, SCC, SCR, and neutral formats
_CASE_CITATION = re.compile(
    r"\b(?:AIR\s+\d{4}\s+(?:SC|[A-Z]{2,4})\s+\d+)"       # AIR 1973 SC 1461
    r"|(?:\(\d{4}\)\s+\d+\s+SCC\s+\d+)"                    # (2023) 5 SCC 123
    r"|(?:\d{4}\s+SCC\s+OnLine\s+SC\s+\d+)"                # 2023 SCC OnLine SC 456
    r"|(?:\[\d{4}\]\s+\d+\s+SCR\s+\d+)"                    # [2023] 1 SCR 45
    r"|(?:\d{4}\s+(?:INSC|INDL)\s+\d+)"                    # 2023 INSC 789
    r"|(?:\b[A-Z][A-Za-z\s\.]+v\.?\s+[A-Z][A-Za-z\s\.]+)",  # Petitioner v. Respondent
    re.IGNORECASE,
)

# Monetary values — Indian format
_MONEY = re.compile(
    r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{1,2})?"
    r"(?:\s*(?:crore|lakh|lac|thousand|million|billion)s?)?",
    re.IGNORECASE,
)

# Dates — multiple Indian formats
_DATE = re.compile(
    r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b"                  # DD/MM/YYYY, DD-MM-YY
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June"
    r"|July|August|September|October|November|December)"
    r"[,\s]+\d{4}\b"                                         # 15th January, 2023
    r"|\b(?:January|February|March|April|May|June"
    r"|July|August|September|October|November|December)"
    r"\s+\d{1,2}[,\s]+\d{4}\b",                              # January 15, 2023
    re.IGNORECASE,
)

# Case numbers — Indian court formats
_CASE_NUMBER = re.compile(
    r"\b(?:W\.?P\.?|SLP|C\.?A\.?|Crl\.?\s*A\.?|M\.?A\.?|O\.?A\.?|T\.?P\.?|R\.?P\.?)"
    r"\s*\(?(?:C|Crl|Civil|Criminal)?\)?"
    r"\s*(?:No\.?\s*)?\d+(?:\s*/\s*\d{2,4})?",
    re.IGNORECASE,
)

# Known Indian courts
_COURT_NAMES = re.compile(
    r"\b(?:Supreme Court of India|Supreme Court|Hon'?ble Supreme Court"
    r"|High Court of [\w\s]+|(?:Bombay|Delhi|Madras|Calcutta|Allahabad|Karnataka"
    r"|Kerala|Gujarat|Andhra Pradesh|Telangana|Rajasthan|Punjab and Haryana"
    r"|Gauhati|Patna|Orissa|Jharkhand|Chhattisgarh|Uttarakhand|Meghalaya"
    r"|Tripura|Sikkim|Manipur|Himachal Pradesh|Jammu and Kashmir"
    r"|Madhya Pradesh) High Court"
    r"|NCLAT|NCLT|NCDRC|SCDRC|DCDRC"
    r"|National Green Tribunal|NGT"
    r"|Central Administrative Tribunal|CAT"
    r"|District (?:and Sessions )?Court"
    r"|Sessions Court|Metropolitan Magistrate"
    r"|Chief Metropolitan Magistrate"
    r"|(?:Additional )?(?:Chief )?Judicial Magistrate"
    r"|Family Court|Labour Court|Consumer Forum"
    r"|Debts Recovery Tribunal|DRT"
    r"|Income Tax Appellate Tribunal|ITAT"
    r"|Customs Excise and Service Tax Appellate Tribunal|CESTAT)\b",
    re.IGNORECASE,
)

# Judge references
_JUDGE_PATTERN = re.compile(
    r"\b(?:(?:Hon'?ble\s+)?(?:(?:Chief\s+)?Justice|J\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
    r"|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3},?\s+(?:JJ?\.?|C\.?J\.?))",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Regex extraction
# --------------------------------------------------------------------------- #
def _extract_regex(text: str) -> list[Entity]:
    """Run all regex patterns and return deduplicated entities."""
    entities: list[Entity] = []
    _seen: set[tuple[str, str]] = set()

    patterns: list[tuple[re.Pattern, str]] = [
        (_STATUTE_SECTION, "STATUTE"),
        (_BARE_STATUTE, "STATUTE"),
        (_SECTION_REF, "PROVISION"),
        (_CASE_CITATION, "CASE_CITATION"),
        (_MONEY, "MONEY"),
        (_DATE, "DATE"),
        (_CASE_NUMBER, "CASE_NUMBER"),
        (_COURT_NAMES, "COURT"),
        (_JUDGE_PATTERN, "JUDGE"),
    ]

    for pat, label in patterns:
        for m in pat.finditer(text):
            t = m.group().strip()
            key = (t.lower(), label)
            if key not in _seen and len(t) > 2:
                _seen.add(key)
                entities.append(Entity(
                    text=t, label=label,
                    start=m.start(), end=m.end(), source="regex"))
    return entities


# --------------------------------------------------------------------------- #
# spaCy extraction (lazy)
# --------------------------------------------------------------------------- #
_nlp = None


def _load_spacy():
    """Load spaCy model lazily. Falls back gracefully if unavailable."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_lg")
        except OSError:
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                _nlp = None
    except ImportError:
        _nlp = None
    return _nlp


# spaCy label → SmartLawAI label mapping
_SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE",
    "MONEY": "MONEY",
    "CARDINAL": None,  # skip
    "ORDINAL": None,   # skip
}


def _extract_spacy(text: str) -> list[Entity]:
    """Extract general NER entities via spaCy. Returns empty if unavailable."""
    nlp = _load_spacy()
    if nlp is None:
        return []

    # Process in chunks if text is long (spaCy max_length default = 1M chars)
    max_len = 100_000
    entities: list[Entity] = []
    _seen: set[tuple[str, str]] = set()

    for start in range(0, len(text), max_len):
        chunk = text[start:start + max_len]
        doc = nlp(chunk)
        for ent in doc.ents:
            label = _SPACY_LABEL_MAP.get(ent.label_)
            if label is None:
                continue
            t = ent.text.strip()
            key = (t.lower(), label)
            if key not in _seen and len(t) > 1:
                _seen.add(key)
                entities.append(Entity(
                    text=t, label=label,
                    start=start + ent.start_char,
                    end=start + ent.end_char,
                    source="spacy"))
    return entities


# --------------------------------------------------------------------------- #
# Merge, deduplicate, normalise
# --------------------------------------------------------------------------- #
def _merge_entities(regex_ents: list[Entity], spacy_ents: list[Entity]) -> list[Entity]:
    """Merge regex and spaCy entities. Regex takes priority for overlapping spans."""
    merged: list[Entity] = list(regex_ents)
    regex_keys = {(e.text.lower(), e.label) for e in regex_ents}

    # Also track regex-covered character spans to avoid duplicates
    regex_spans = set()
    for e in regex_ents:
        if e.start >= 0 and e.end >= 0:
            regex_spans.update(range(e.start, e.end))

    for e in spacy_ents:
        key = (e.text.lower(), e.label)
        if key in regex_keys:
            continue
        # Skip spaCy entities that overlap with regex-covered spans
        if e.start >= 0 and e.end >= 0:
            ent_range = set(range(e.start, e.end))
            if ent_range & regex_spans:
                continue
        merged.append(e)

    return merged


def _classify_into_buckets(entities: list[Entity]) -> dict[str, list[str]]:
    """Sort entities into named buckets for the NERResult."""
    buckets: dict[str, list[str]] = {
        "statutes": [], "case_citations": [], "courts": [], "judges": [],
        "parties": [], "dates": [], "monetary_values": [], "legal_provisions": [],
        "organizations": [], "locations": [],
    }
    seen_per_bucket: dict[str, set[str]] = {k: set() for k in buckets}

    label_to_bucket = {
        "STATUTE": "statutes",
        "CASE_CITATION": "case_citations",
        "COURT": "courts",
        "JUDGE": "judges",
        "PERSON": "parties",
        "DATE": "dates",
        "MONEY": "monetary_values",
        "PROVISION": "legal_provisions",
        "CASE_NUMBER": "case_citations",
        "ORGANIZATION": "organizations",
        "LOCATION": "locations",
    }

    for e in entities:
        bucket = label_to_bucket.get(e.label)
        if bucket is None:
            continue
        norm = e.text.strip()
        if norm.lower() not in seen_per_bucket[bucket]:
            seen_per_bucket[bucket].add(norm.lower())
            buckets[bucket].append(norm)

    return buckets


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_entities(doc_id: str, text: str) -> NERResult:
    """Extract all named entities from Indian legal text.

    Uses regex for structured legal patterns and spaCy for general entities.
    Returns a unified NERResult with deduplicated, categorised entities.
    """
    regex_ents = _extract_regex(text)
    spacy_ents = _extract_spacy(text)
    merged = _merge_entities(regex_ents, spacy_ents)
    buckets = _classify_into_buckets(merged)

    return NERResult(
        doc_id=doc_id,
        entities=merged,
        statutes=buckets["statutes"],
        case_citations=buckets["case_citations"],
        courts=buckets["courts"],
        judges=buckets["judges"],
        parties=buckets["parties"],
        dates=buckets["dates"],
        monetary_values=buckets["monetary_values"],
        legal_provisions=buckets["legal_provisions"],
        organizations=buckets["organizations"],
        locations=buckets["locations"],
    )
