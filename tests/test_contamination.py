"""Tests for baselines/contamination.py (T1)."""
from __future__ import annotations

from datetime import date

import pytest


def test_resolve_judgment_date_refuses_to_guess_without_header_signal():
    """Positive test: the heuristic must correctly refuse to guess. This text
    has years embedded (2019, 2020) but none of them are the judgment's own
    stated decision date -- resolve_judgment_date must not mistake a cited
    year for the judgment's own date."""
    from baselines.contamination import resolve_judgment_date

    text = ("This document discusses the agreement made in 2019 and a payment "
            "schedule referencing invoices from 2020, but never states when "
            "the judgment itself was delivered, dated, decided, or pronounced.")
    result = resolve_judgment_date("doc-x", text)
    assert result.source == "unavailable"
    assert result.decision_date is None


def test_resolve_judgment_date_finds_header_phrase():
    from baselines.contamination import resolve_judgment_date

    text = "This judgment was delivered on 15th January, 2023. Further text follows."
    result = resolve_judgment_date("doc-y", text)
    assert result.source == "header_heuristic"
    assert result.decision_date == date(2023, 1, 1)


def test_resolve_judgment_date_only_scans_the_header_window():
    """A 'dated' phrase deep in the body (past _HEADER_SCAN_CHARS) must not be
    picked up -- it's far more likely to belong to a quoted precedent than the
    judgment's own metadata."""
    from baselines.contamination import _HEADER_SCAN_CHARS, resolve_judgment_date

    padding = "x" * (_HEADER_SCAN_CHARS + 100)
    text = padding + " This order was dated 5th May, 2021."
    result = resolve_judgment_date("doc-z", text)
    assert result.source == "unavailable"


def test_split_by_cutoff_raises_for_unconfigured_model():
    from baselines.contamination import DateResolution, split_by_cutoff

    with pytest.raises(KeyError):
        split_by_cutoff([DateResolution("d1", date(2020, 1, 1), "header_heuristic")],
                        "no-such-model")


def test_split_by_cutoff_buckets_correctly(monkeypatch):
    from baselines import config
    from baselines.contamination import DateResolution, split_by_cutoff

    monkeypatch.setitem(config.MODEL_CUTOFF_DATES, "test-model", date(2022, 1, 1))
    resolutions = [
        DateResolution("pre", date(2020, 1, 1), "header_heuristic"),
        DateResolution("post", date(2023, 1, 1), "header_heuristic"),
        DateResolution("undated", None, "unavailable"),
    ]
    split = split_by_cutoff(resolutions, "test-model")
    assert [r.doc_id for r in split["pre_cutoff"]] == ["pre"]
    assert [r.doc_id for r in split["post_cutoff"]] == ["post"]
    assert [r.doc_id for r in split["undated_skipped"]] == ["undated"]
