"""Tests for the Indian Legal NER module (regex + spaCy hybrid)."""
from __future__ import annotations

from smartlawai.core.ner import Entity, extract_entities, _extract_regex


# --------------------------------------------------------------------------- #
# Regex extraction tests
# --------------------------------------------------------------------------- #
def test_statute_section_references():
    text = "Section 302 of Indian Penal Code and Section 420 of IPC apply here."
    ents = _extract_regex(text)
    labels = {e.label for e in ents}
    assert "STATUTE" in labels or "PROVISION" in labels


def test_bare_statute():
    text = "The Indian Contract Act, 1872 governs this matter."
    ents = _extract_regex(text)
    statutes = [e for e in ents if e.label == "STATUTE"]
    assert len(statutes) >= 1
    assert any("Indian Contract Act" in e.text for e in statutes)


def test_new_criminal_codes():
    text = "Under the Bharatiya Nyaya Sanhita (BNS) and Bharatiya Nagarik Suraksha Sanhita (BNSS)"
    ents = _extract_regex(text)
    statute_texts = [e.text for e in ents if e.label == "STATUTE"]
    assert any("BNS" in t or "Bharatiya Nyaya Sanhita" in t for t in statute_texts)


def test_section_references():
    text = "Article 21 of the Constitution guarantees life and liberty. Section 125 CrPC provides maintenance."
    ents = _extract_regex(text)
    provisions = [e for e in ents if e.label == "PROVISION"]
    assert len(provisions) >= 1


def test_air_citation():
    text = "In AIR 1973 SC 1461, the Supreme Court held..."
    ents = _extract_regex(text)
    citations = [e for e in ents if e.label == "CASE_CITATION"]
    assert len(citations) >= 1
    assert any("AIR 1973 SC 1461" in e.text for e in citations)


def test_scc_citation():
    text = "As per (2023) 5 SCC 123, the ratio decidendi was clear."
    ents = _extract_regex(text)
    citations = [e for e in ents if e.label == "CASE_CITATION"]
    assert len(citations) >= 1


def test_monetary_values():
    text = "The damages awarded were ₹50,00,000 and Rs. 25 lakh."
    ents = _extract_regex(text)
    money = [e for e in ents if e.label == "MONEY"]
    assert len(money) >= 2


def test_dates():
    text = "The judgment was delivered on 15th January, 2023. Filed on 01/03/2022."
    ents = _extract_regex(text)
    dates = [e for e in ents if e.label == "DATE"]
    assert len(dates) >= 2


def test_case_numbers():
    text = "In W.P.(C) No. 1234/2023 before the Delhi High Court."
    ents = _extract_regex(text)
    case_nums = [e for e in ents if e.label == "CASE_NUMBER"]
    assert len(case_nums) >= 1


def test_court_names():
    text = "The Supreme Court of India and the Bombay High Court heard arguments."
    ents = _extract_regex(text)
    courts = [e for e in ents if e.label == "COURT"]
    assert len(courts) >= 2


def test_tribunal_names():
    text = "NCLT Mumbai ordered liquidation. The NCLAT upheld the order."
    ents = _extract_regex(text)
    courts = [e for e in ents if e.label == "COURT"]
    assert len(courts) >= 2


def test_judge_patterns():
    text = "Hon'ble Justice D.Y. Chandrachud delivered the judgment."
    ents = _extract_regex(text)
    judges = [e for e in ents if e.label == "JUDGE"]
    assert len(judges) >= 1


# --------------------------------------------------------------------------- #
# Full pipeline tests
# --------------------------------------------------------------------------- #
def test_extract_entities_integration():
    """Test the full extract_entities pipeline with a realistic legal snippet."""
    text = """
    IN THE SUPREME COURT OF INDIA
    CIVIL APPEAL No. 1234/2023
    Ram Kumar v. State of Maharashtra
    Date: 15th March, 2023

    The appellant was convicted under Section 302 of the Indian Penal Code, 1860.
    The Bombay High Court upheld the conviction. The fine of ₹5,00,000 was confirmed.
    Hon'ble Justice D.Y. Chandrachud delivered the judgment.
    Reference: AIR 1973 SC 1461 and (2019) 4 SCC 567.
    """
    result = extract_entities("test-doc-1", text)

    assert result.doc_id == "test-doc-1"
    assert len(result.entities) > 0
    assert len(result.courts) >= 1  # Supreme Court and/or Bombay High Court
    assert len(result.monetary_values) >= 1
    assert len(result.dates) >= 1
    assert len(result.legal_provisions) >= 1 or len(result.statutes) >= 1
    assert len(result.case_citations) >= 1


def test_deduplication():
    """Entities should be deduplicated by (text.lower(), label)."""
    text = "Section 302 IPC was applied. Section 302 IPC was confirmed."
    result = extract_entities("dedup-test", text)
    # The same entity mentioned twice should appear only once
    provision_texts = [e.text.lower() for e in result.entities if e.label == "PROVISION"]
    statute_texts = [e.text.lower() for e in result.entities if e.label == "STATUTE"]
    all_texts = provision_texts + statute_texts
    assert len(all_texts) == len(set(all_texts))


def test_empty_text():
    """Empty text should return empty results without errors."""
    result = extract_entities("empty-doc", "")
    assert result.doc_id == "empty-doc"
    assert len(result.entities) == 0


def test_entity_schema():
    """Entity dataclass should have all required fields."""
    e = Entity(text="IPC", label="STATUTE", start=0, end=3, source="regex")
    assert e.text == "IPC"
    assert e.label == "STATUTE"
    assert e.start == 0
    assert e.end == 3
    assert e.source == "regex"
