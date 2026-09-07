"""M4 authority registry tests.

`test_repealed_ipc_section_is_struck` is PLAN.md M4's stated done-when
condition and must not be deleted or skipped.
"""

from __future__ import annotations

from datetime import date

import pytest

from registry.normalise import (
    is_compatible,
    normalise_act,
    parse_section_ref,
    section_id,
)
from registry.resolver import resolve_citation
from registry.types import (
    STATUS_IN_FORCE,
    STATUS_NOT_FOUND,
    STATUS_REPEALED,
    STATUS_SUPERSEDED,
    TARGET_SECTION,
    TARGET_STATUTE,
    ResolvedCitation,
)
from verify.registry_check import check_citation

TODAY = date(2026, 9, 5)
BEFORE_COMMENCEMENT = date(2020, 1, 1)
COMMENCEMENT = date(2024, 7, 1)


# --------------------------------------------------------------------------- #
# THE done-when test (PLAN.md M4)
# --------------------------------------------------------------------------- #
def test_repealed_ipc_section_is_struck():
    """A claim citing IPC 420 as of today is struck and annotated.

    PLAN.md M4: *"a claim citing IPC 420 with as-of date today is struck and
    annotated 'repealed 2024-07-01; see BNS 318'."*
    """
    cit = resolve_citation("Section 420 IPC")
    assert cit is not None, "IPC 420 must resolve against the registry"
    assert cit.target_type == TARGET_SECTION
    assert cit.target_id == "ipc-1860-s420"

    res = check_citation(cit, as_of=TODAY)

    # Struck: gate.py strikes anything that is not in_force.
    assert res.status != STATUS_IN_FORCE
    assert res.status == STATUS_SUPERSEDED
    # Annotated with the repeal date and the successor provision.
    assert res.note is not None
    assert "repealed 2024-07-01" in res.note
    assert "BNS 318" in res.note
    assert res.as_of == TODAY


# --------------------------------------------------------------------------- #
# Temporal correctness
# --------------------------------------------------------------------------- #
def test_ipc_section_was_in_force_before_commencement():
    """The same citation is valid law as of a pre-2024 date."""
    cit = resolve_citation("Section 420 IPC")
    res = check_citation(cit, as_of=BEFORE_COMMENCEMENT)
    assert res.status == STATUS_IN_FORCE
    assert res.note is None


def test_repeal_boundary_is_inclusive_on_commencement_day():
    """On 2024-07-01 itself the old code is already gone."""
    cit = resolve_citation("Section 420 IPC")
    assert check_citation(cit, as_of=COMMENCEMENT).status != STATUS_IN_FORCE
    day_before = date(2024, 6, 30)
    assert check_citation(cit, as_of=day_before).status == STATUS_IN_FORCE


def test_successor_section_is_in_force_today():
    """BNS 318 is current law."""
    cit = resolve_citation("Section 318 BNS")
    assert cit is not None
    assert check_citation(cit, as_of=TODAY).status == STATUS_IN_FORCE


def test_successor_not_in_force_before_its_commencement():
    """BNS did not exist in 2020; asking about it then is not 'in force'."""
    cit = resolve_citation("Section 318 BNS")
    res = check_citation(cit, as_of=BEFORE_COMMENCEMENT)
    assert res.status != STATUS_IN_FORCE


# --------------------------------------------------------------------------- #
# Repealed with no successor -- the sedition case
# --------------------------------------------------------------------------- #
def test_sedition_is_repealed_with_no_successor():
    """IPC 124A was deleted outright; BNS 152 is a different offence.

    The official correspondence table records "Deleted" against 124A. The
    registry must not redirect it to a successor it does not have.
    """
    cit = resolve_citation("Section 124A IPC")
    assert cit is not None
    res = check_citation(cit, as_of=TODAY)
    assert res.status == STATUS_REPEALED
    assert "no corresponding provision" in (res.note or "")


def test_unmapped_repealed_section_does_not_claim_there_is_no_successor():
    """"We did not parse a mapping" must not be reported as "there is none".

    IPC 337 is a real provision with a BNS counterpart, but its correspondence
    row did not parse as a mapping, so the registry holds no successor for it.
    Saying "no corresponding provision" would assert a fact we have not
    established (CLAUDE.md I2). The claim is struck either way; the annotation
    must stay honest about which of the two situations this is.
    """
    cit = resolve_citation("Section 337 IPC")
    assert cit is not None
    res = check_citation(cit, as_of=TODAY)
    assert res.status != STATUS_IN_FORCE
    assert "no corresponding provision" not in (res.note or "")
    assert "not recorded" in (res.note or "")


# --------------------------------------------------------------------------- #
# I2: never fail open, never guess
# --------------------------------------------------------------------------- #
def test_unknown_section_resolves_to_none_not_a_guess():
    """A syntactically valid but nonexistent section is unresolved, not invented."""
    assert resolve_citation("Section 9999 IPC") is None


def test_structurally_impossible_citation_is_refused():
    """The IPC has no Articles; do not silently coerce to a section."""
    assert resolve_citation("Article 21 IPC") is None


def test_unknown_act_resolves_to_none():
    assert resolve_citation("Section 5 of the Made Up Act 2019") is None
    assert resolve_citation("") is None


def test_check_never_raises_on_garbage():
    """check_citation must return not_found, never propagate an exception."""
    bogus = ResolvedCitation(raw="x", target_type=TARGET_SECTION,
                             target_id="no-such-section", confidence=1.0)
    res = check_citation(bogus, as_of=TODAY)
    assert res.status == STATUS_NOT_FOUND
    assert res.as_of == TODAY


def test_check_handles_none_citation_without_raising():
    res = check_citation(None, as_of=TODAY)
    assert res.status == STATUS_NOT_FOUND


# --------------------------------------------------------------------------- #
# Statute-level targets
# --------------------------------------------------------------------------- #
def test_bare_act_resolves_to_statute_and_is_superseded():
    cit = resolve_citation("Indian Penal Code")
    assert cit is not None
    assert cit.target_type == TARGET_STATUTE
    assert cit.target_id == "ipc-1860"
    res = check_citation(cit, as_of=TODAY)
    assert res.status == STATUS_SUPERSEDED
    assert "BNS" in (res.note or "")


def test_constitution_article_is_in_force():
    """The gap this previously guarded is closed: articles are now ingested."""
    cit = resolve_citation("Article 21 of the Constitution of India")
    assert cit is not None
    assert cit.target_id == "constitution-1950-art21"
    assert check_citation(cit, as_of=TODAY).status == STATUS_IN_FORCE


def test_article_31_repeal_boundary_is_dated():
    """Article 31's omission date was recovered, so it has a real interval.

    This is a *second* dated natural experiment alongside IPC->BNS: Article 31
    (right to property) was omitted by the 44th Amendment with effect from
    1979-06-20, and it is cited 443 times in the corpus. A pre-1979 judgment
    citing it was right at the time and is wrong for a present-day query --
    which is the project's whole thesis, reproduced in a second statute.
    """
    cit = resolve_citation("Article 31 of the Constitution of India")
    assert cit is not None
    assert check_citation(cit, as_of=date(1960, 1, 1)).status == STATUS_IN_FORCE
    assert check_citation(cit, as_of=date(1979, 6, 19)).status == STATUS_IN_FORCE
    res = check_citation(cit, as_of=date(1979, 6, 20))
    assert res.status != STATUS_IN_FORCE
    assert "1979-06-20" in (res.note or "")
    assert check_citation(cit, as_of=TODAY).status != STATUS_IN_FORCE


def test_undated_omission_is_still_struck_at_every_date():
    """Where the source states no date, we still refuse to claim in-force.

    Article 238's omission date is not in the official text at all -- it appears
    only in the arrangement of articles as "[238. Omitted.]". Without a date we
    cannot say it was in force in 1960 either, so it is struck throughout and
    the note says exactly why.
    """
    cit = resolve_citation("Article 238 of the Constitution")
    assert cit is not None
    for d in (date(1960, 1, 1), date(1990, 1, 1), TODAY):
        res = check_citation(cit, as_of=d)
        assert res.status != STATUS_IN_FORCE
        assert "not recorded" in (res.note or "")


# --------------------------------------------------------------------------- #
# Normalisation invariants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,act,sid",
    [
        ("Section 420 IPC", "ipc-1860", "ipc-1860-s420"),
        ("Section 302 of the Indian Penal Code", "ipc-1860", "ipc-1860-s302"),
        ("S. 302A(1)(b) IPC", "ipc-1860", "ipc-1860-s302A"),
        ("Section 138 of the Negotiable Instruments Act", "ni-1881", "ni-1881-s138"),
        ("Article 21 of the Constitution of India", "constitution-1950",
         "constitution-1950-art21"),
    ],
)
def test_normalisation(raw, act, sid):
    a = normalise_act(raw)
    r = parse_section_ref(raw)
    assert a == act
    assert section_id(a, r) == sid


def test_bns_and_bnss_are_not_confused():
    assert normalise_act("BNS") == "bns-2023"
    assert normalise_act("BNSS") == "bnss-2023"
    assert normalise_act("Cr.P.C.") == "crpc-1973"


def test_act_year_is_never_read_as_a_section():
    """"Indian Penal Code, 1860" must not become section 1860."""
    assert parse_section_ref("Indian Penal Code, 1860") is None
    assert parse_section_ref("Negotiable Instruments Act 1881") is None


def test_kind_compatibility_rules():
    assert is_compatible("constitution-1950", "art")
    assert not is_compatible("ipc-1860", "art")
    assert not is_compatible("constitution-1950", "s")
    assert is_compatible("ipc-1860", "s")
    assert is_compatible("cpc-1908", "o")
