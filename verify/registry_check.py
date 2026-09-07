"""Authority check: was this citation in force on a given date?

Public contract (frozen -- M4_ONBOARDING.md 6, consumed by M5's `gate.py`):

    check_citation(citation: ResolvedCitation, as_of: date) -> RegistryResult

**This function never raises.** A safety component that cannot answer returns
`not_found`; it does not fail open and it does not substitute a default
(CLAUDE.md I2 -- the old code returning 0.5 is exactly the bug this forbids).
`gate.py` treats every non-`in_force` status identically: it strikes the claim.

Status semantics:

| status       | meaning                                                    |
|--------------|------------------------------------------------------------|
| `in_force`   | in force on `as_of`                                        |
| `superseded` | no longer in force, and a successor provision is recorded  |
| `repealed`   | no longer in force, with no successor                      |
| `not_found`  | not in the registry, or the registry could not be read     |
"""

from __future__ import annotations

from datetime import date

from registry.normalise import abbrev
from registry.resolver import _db
from registry.types import (
    STATUS_IN_FORCE,
    STATUS_NOT_FOUND,
    STATUS_REPEALED,
    STATUS_SUPERSEDED,
    TARGET_SECTION,
    RegistryResult,
    ResolvedCitation,
)


def _as_date(v) -> date | None:
    if v is None:
        return None
    return v if isinstance(v, date) else date.fromisoformat(str(v)[:10])


def _successors(conn, section_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT to_section_id FROM SUPERSESSION "
        "WHERE from_section_id = ? AND to_section_id IS NOT NULL "
        "ORDER BY to_section_id",
        [section_id],
    ).fetchall()
    return [r[0] for r in rows]


def _explicitly_no_successor(conn, section_id: str) -> bool:
    """Did a source explicitly record this provision as having no counterpart?

    This distinguishes two very different situations that both present as
    "zero successors":

    * the correspondence table recorded the provision as *Deleted* -- we know
      there is no successor (IPC 124A, sedition); versus
    * no mapping row was parsed for it at all -- we do not know either way.

    Reporting the second as "no corresponding provision" would assert a fact we
    have not established, which is exactly what CLAUDE.md I2 forbids. Both still
    strike the claim; only the annotation differs.
    """
    row = conn.execute(
        "SELECT 1 FROM SUPERSESSION "
        "WHERE from_section_id = ? AND relation = 'repealed_no_successor' LIMIT 1",
        [section_id],
    ).fetchone()
    return row is not None


def _render(section_id: str) -> str:
    """'bns-2023-s318' -> 'BNS 318'."""
    if "-s" in section_id:
        act, num = section_id.rsplit("-s", 1)
        return f"{abbrev(act)} {num}"
    return section_id


def check_citation(citation: ResolvedCitation, as_of: date, db=None) -> RegistryResult:
    """Never raises. Returns `not_found` rather than guessing."""
    try:
        if citation is None or not getattr(citation, "target_id", None):
            return RegistryResult(STATUS_NOT_FOUND, as_of, "no target id")

        conn = db if db is not None else _db()

        if citation.target_type == TARGET_SECTION:
            row = conn.execute(
                "SELECT id, statute_id, in_force_from, in_force_to, status "
                "FROM SECTIONS WHERE id = ?", [citation.target_id]
            ).fetchone()
            if row is None:
                return RegistryResult(
                    STATUS_NOT_FOUND, as_of,
                    f"{citation.target_id} not in registry")
            _sid, statute_id, iff, ift, status = row
            iff, ift = _as_date(iff), _as_date(ift)

            # A provision recorded as omitted with no effective date. We know it
            # was removed but not when, so we cannot say whether it was in force
            # on an arbitrary `as_of`. Reporting it as in force would be wrong;
            # inventing a date would be worse. Strike it and say exactly that.
            if status == "omitted" and ift is None:
                succ = _successors(conn, citation.target_id)
                note = "omitted from the %s; effective date not recorded in registry" % (
                    "Constitution" if statute_id == "constitution-1950" else "statute")
                if succ:
                    note += "; see " + " / ".join(_render(s) for s in succ)
                    return RegistryResult(STATUS_SUPERSEDED, as_of, note)
                return RegistryResult(STATUS_REPEALED, as_of, note)

            # A section inherits its statute's dates when it carries none of
            # its own. Both directions matter: without the commencement date,
            # BNS 318 reports "in force" as of 2020 -- four years before the
            # BNS existed.
            if ift is None or iff is None:
                srow = conn.execute(
                    "SELECT in_force_from, repealed_on FROM STATUTES WHERE id = ?",
                    [statute_id],
                ).fetchone()
                if srow:
                    if iff is None:
                        iff = _as_date(srow[0])
                    if ift is None:
                        ift = _as_date(srow[1])

            if iff is not None and as_of < iff:
                return RegistryResult(
                    STATUS_NOT_FOUND, as_of,
                    f"not in force until {iff.isoformat()}")

            if ift is not None and as_of >= ift:
                succ = _successors(conn, citation.target_id)
                if succ:
                    return RegistryResult(
                        STATUS_SUPERSEDED, as_of,
                        "repealed {}; see {}".format(
                            ift.isoformat(),
                            " / ".join(_render(s) for s in succ)),
                    )
                if _explicitly_no_successor(conn, citation.target_id):
                    return RegistryResult(
                        STATUS_REPEALED, as_of,
                        f"repealed {ift.isoformat()}; no corresponding provision")
                return RegistryResult(
                    STATUS_REPEALED, as_of,
                    f"repealed {ift.isoformat()}; successor not recorded in registry")

            return RegistryResult(STATUS_IN_FORCE, as_of, None)

        # --- statute-level target ---
        row = conn.execute(
            "SELECT id, in_force_from, repealed_on, repealed_by_statute_id "
            "FROM STATUTES WHERE id = ?", [citation.target_id]
        ).fetchone()
        if row is None:
            return RegistryResult(
                STATUS_NOT_FOUND, as_of,
                f"{citation.target_id} not in registry")
        _sid, iff, rep, repby = row
        iff, rep = _as_date(iff), _as_date(rep)

        if iff is not None and as_of < iff:
            return RegistryResult(
                STATUS_NOT_FOUND, as_of,
                f"not in force until {iff.isoformat()}")
        if rep is not None and as_of >= rep:
            if repby:
                return RegistryResult(
                    STATUS_SUPERSEDED, as_of,
                    f"repealed {rep.isoformat()}; see {abbrev(repby)}")
            return RegistryResult(
                STATUS_REPEALED, as_of, f"repealed {rep.isoformat()}")
        return RegistryResult(STATUS_IN_FORCE, as_of, None)

    except Exception as exc:  # noqa: BLE001 -- I2: never fail open
        return RegistryResult(
            STATUS_NOT_FOUND, as_of, f"registry unavailable: {type(exc).__name__}")
