"""Tests for PII redaction and entity constraints (PLAN.md M5.7)."""

from __future__ import annotations

from smartlawai.guardrails.pii import (
    ALLOWED_PII_ENTITIES,
    redact,
    redact_pii,
)


def test_pii_allowed_entities_strictly_excludes_person_location_date():
    """PLAN.md M5.7: Never PERSON, LOCATION, or DATE_TIME."""
    assert "PERSON" not in ALLOWED_PII_ENTITIES
    assert "LOCATION" not in ALLOWED_PII_ENTITIES
    assert "DATE_TIME" not in ALLOWED_PII_ENTITIES
    assert set(ALLOWED_PII_ENTITIES) == {
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "CREDIT_CARD",
        "IN_PAN",
        "IN_AADHAAR",
    }


def test_pii_redacts_indian_pan_and_email():
    text = "Assessee PAN is ABCDE1234F and email is advocate@delhicourt.nic.in for proceedings."
    res = redact_pii(text)
    assert res.status == "ok"
    assert res.redacted_count == 2
    assert "ABCDE1234F" not in res.redacted_text
    assert "<IN_PAN>" in res.redacted_text
    assert "advocate@delhicourt.nic.in" not in res.redacted_text
    assert "<EMAIL_ADDRESS>" in res.redacted_text


def test_pii_preserves_judge_names_dates_and_locations():
    """Judgment substance must remain completely intact."""
    legal_text = (
        "On 15 January 2024, Hon'ble Justice D.Y. Chandrachud sitting at New Delhi "
        "delivered the judgment in the Supreme Court of India."
    )
    res = redact_pii(legal_text)
    assert res.status == "ok"
    assert res.redacted_count == 0
    assert res.redacted_text == legal_text
    assert "Justice D.Y. Chandrachud" in res.redacted_text
    assert "New Delhi" in res.redacted_text
    assert "15 January 2024" in res.redacted_text


def test_pii_ordering_stores_spans_separately_before_redaction():
    text = "Contact at +91 9876543210 immediately."
    res = redact_pii(text)
    assert len(res.spans) == 1
    span = res.spans[0]
    assert span.entity_type == "PHONE_NUMBER"
    assert "9876543210" in span.value
    assert span.start >= 0
    assert span.end > span.start
    assert "<PHONE_NUMBER>" in res.redacted_text


def test_pii_backward_compatible_redact_tuple():
    text = "Email test@example.com"
    redacted_str, count = redact(text)
    assert count == 1
    assert "<EMAIL_ADDRESS>" in redacted_str
