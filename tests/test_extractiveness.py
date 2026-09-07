"""Extractiveness tests (M6 Phase 7) -- RESEARCH.md §5.

Mandatory alongside faithfulness: HHEM rewards verbatim copying, so a model that
quotes the source scores near-perfect faithfulness while being a useless
summariser. These tests pin the behaviour that makes the number meaningful.
"""
from __future__ import annotations

import pytest

from eval import config
from eval.extractiveness import (
    coverage,
    extractiveness,
    extractiveness_report,
    longest_copied_span,
    mean_extractiveness,
)

SOURCE = ("The appellant was granted quasi permanent allotment of lands in the "
          "village of Raikot in Ludhiana District in the year 1949.")


def test_verbatim_copy_scores_one():
    """The copier case the metric exists to expose."""
    assert extractiveness(SOURCE, SOURCE) == 1.0


def test_copied_substring_scores_one():
    assert extractiveness("granted quasi permanent allotment of lands", SOURCE) == 1.0


def test_disjoint_text_scores_zero():
    assert extractiveness("entirely different words appear nowhere here", SOURCE) == 0.0


def test_partial_copy_is_between():
    score = extractiveness("The appellant was granted a pension by the State",
                           SOURCE)
    assert 0.0 < score < 1.0


def test_default_ngram_is_four():
    assert config.EXTRACTIVENESS_NGRAM == 4


def test_prediction_shorter_than_n_is_unavailable_not_zero():
    """A three-word answer has not been shown to be original; it has not been
    measured (I2 applied to the metric's own input)."""
    assert extractiveness("too short", SOURCE) is None
    assert extractiveness("one two three", SOURCE, n=4) is None
    assert extractiveness("one two three", SOURCE, n=3) is not None


def test_empty_prediction_is_unavailable():
    assert extractiveness("", SOURCE) is None


def test_empty_source_scores_zero_not_none():
    """Nothing could have been copied -- that is a measurement, not a gap."""
    assert extractiveness("some words that are long enough", "") == 0.0


def test_punctuation_and_case_do_not_defeat_the_metric():
    """Copying a sentence and changing its comma is still copying."""
    assert extractiveness("GRANTED, QUASI: PERMANENT; ALLOTMENT", SOURCE) == 1.0


def test_smaller_ngram_is_more_permissive():
    text = "The appellant was awarded costs in the village court"
    assert extractiveness(text, SOURCE, n=2) >= extractiveness(text, SOURCE, n=4)


def test_zero_or_negative_ngram_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        extractiveness("a b c d", SOURCE, n=0)


# --------------------------------------------------------------------------- #
# Coverage and longest span -- the two numbers reported next to it
# --------------------------------------------------------------------------- #
def test_coverage_is_unigram_overlap():
    assert coverage("appellant lands village", SOURCE) == 1.0
    assert coverage("zebra giraffe", SOURCE) == 0.0


def test_paraphrase_has_high_coverage_but_low_extractiveness():
    """The signature of a genuine summary rather than a copy."""
    paraphrase = "In 1949 the village lands were allotted to the appellant"
    assert coverage(paraphrase, SOURCE) > 0.5
    assert extractiveness(paraphrase, SOURCE) < 0.5


def test_coverage_of_empty_prediction_is_unavailable():
    assert coverage("", SOURCE) is None


def test_longest_copied_span_finds_the_run():
    assert longest_copied_span("granted quasi permanent allotment", SOURCE) == 4


def test_longest_copied_span_distinguishes_one_long_quote_from_scattered_words():
    scattered = "appellant zzz lands zzz village zzz district"
    quoted = "in the village of Raikot in Ludhiana"
    assert longest_copied_span(quoted, SOURCE) > longest_copied_span(scattered, SOURCE)


def test_longest_copied_span_of_empty_input_is_zero():
    assert longest_copied_span("", SOURCE) == 0
    assert longest_copied_span(SOURCE, "") == 0


# --------------------------------------------------------------------------- #
# Report and corpus aggregate
# --------------------------------------------------------------------------- #
def test_report_bundles_the_three_numbers():
    r = extractiveness_report(SOURCE, SOURCE)
    assert r.extractiveness == 1.0
    assert r.coverage == 1.0
    assert r.longest_copied_span > 0
    assert r.status == "ok"
    assert r.ngram == config.EXTRACTIVENESS_NGRAM


def test_report_states_why_a_short_prediction_is_unavailable():
    r = extractiveness_report("two words", SOURCE)
    assert r.status == "unavailable"
    assert "shorter_than_4_tokens" in r.reason
    assert r.to_dict()["extractiveness"] is None


def test_mean_skips_pairs_too_short_to_measure():
    """Otherwise a system that refuses often would look maximally original."""
    preds = ["granted quasi permanent allotment", "no"]
    assert mean_extractiveness(preds, [SOURCE, SOURCE]) == 1.0


def test_mean_of_all_unmeasurable_pairs_is_none():
    assert mean_extractiveness(["no", "hi"], [SOURCE, SOURCE]) is None


def test_mean_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="align"):
        mean_extractiveness(["a b c d e"], [SOURCE, SOURCE])
