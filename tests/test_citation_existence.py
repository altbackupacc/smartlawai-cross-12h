"""Tests for baselines/citation_existence.py, against sample_docs/
sample_judgment.txt (contains 'Section 74 of the Indian Contract Act, 1872'
and 'Arbitration and Conciliation Act, 1996' -- real, extractable citations)."""
from __future__ import annotations

import os
import tempfile
from unittest import mock


def _backend(tmp):
    os.environ["SMARTLAW_LOCAL_DIR"] = tmp
    from smartlawai.adapters.factory import get_backend
    return get_backend("local")


def _ingest_sample_judgment(be, owner_id="alice"):
    from smartlawai.pipeline import Pipeline
    here = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(here, "sample_docs", "sample_judgment.txt")
    doc_id, _ = Pipeline(be).ingest(path, doc_type="JUDGMENT", source="test",
                                    owner_id=owner_id)
    return doc_id


def test_existing_citation_resolves_true():
    from baselines.citation_existence import build_corpus_citation_index, check_existence
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        doc_id = _ingest_sample_judgment(be)
        index = build_corpus_citation_index(be, Scope(doc_ids=(doc_id,), owner_id="alice"))

        result = check_existence("Section 74 of the Indian Contract Act, 1872", index)
        assert result.status == "ok"
        assert result.exists_in_corpus is True


def test_fabricated_citation_resolves_false():
    from baselines.citation_existence import build_corpus_citation_index, check_existence
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        doc_id = _ingest_sample_judgment(be)
        index = build_corpus_citation_index(be, Scope(doc_ids=(doc_id,), owner_id="alice"))

        result = check_existence("Section 999 of the Made-Up Act, 2099", index)
        assert result.status == "ok"
        assert result.exists_in_corpus is False


def test_missing_index_is_unavailable_not_a_guess():
    """I2-style discipline: no index -> status='unavailable', never a
    silently-substituted True/False."""
    from baselines.citation_existence import check_existence

    result = check_existence("anything", None)
    assert result.status == "unavailable"
    assert result.exists_in_corpus is None


def test_build_corpus_citation_index_never_calls_fetch_all_chunks():
    from baselines.citation_existence import build_corpus_citation_index
    from smartlawai.scope import Scope

    with tempfile.TemporaryDirectory() as tmp:
        be = _backend(tmp)
        doc_id = _ingest_sample_judgment(be)
        with mock.patch.object(type(be), "fetch_all_chunks",
                               side_effect=AssertionError("I1 violation: fetch_all_chunks called")):
            build_corpus_citation_index(be, Scope(doc_ids=(doc_id,), owner_id="alice"))


def test_to_registry_result_dict_matches_real_m4_m6_shape():
    """eval/metrics.py (M6, now merged) reads a "status" key with one of M4's
    real RegistryResult values -- not the {"valid", "in_force"} shape this
    function originally guessed before M4/M6 existed. An existing citation
    maps to "in_force" (safe input for citation_validity_rate/
    registry_coverage_rate, which don't distinguish which resolved status
    applies) with a note stating the baseline performed no in-force check --
    never a bare in_force claim with no caveat attached."""
    from baselines.citation_existence import CitationExistenceResult, to_registry_result_dict

    d = to_registry_result_dict(
        CitationExistenceResult("Section 302 IPC", status="ok", exists_in_corpus=True))
    assert d["status"] == "in_force"
    assert d["note"] is not None and "no in-force" in d["note"]

    d2 = to_registry_result_dict(
        CitationExistenceResult("Section 9999 IPC", status="ok", exists_in_corpus=False))
    assert d2 == {"status": "not_found", "as_of": None, "note": None}
