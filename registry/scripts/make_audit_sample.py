"""Draw the verification samples PLAN.md M4 requires, and write worksheets.

Two disjoint samples:

* **verify_100.csv** — the stratified 100-entry hand-check against the Gazette
  (PLAN.md M4 step 3). Its result is the "parse verification accuracy" the paper
  reports.
* **audit_200.csv** — the independent 200-entry audit (threat T10), which must
  be done by someone who did not build the registry.

Stratification is deliberate, not uniform:

1. **Source strength.** IPC/CrPC/IEA section numbers rest on a *single* source
   (the correspondence tables); every other statute was cross-validated against
   India Code. Single-sourced rows carry more risk and are over-sampled.
2. **Relation type.** `repealed_no_successor` and one-to-many mappings are rarer
   than plain 1:1 but are where parsing errors concentrate, so each is
   guaranteed a share rather than left to chance.
3. **Citation frequency.** A wrong mapping on a section cited 400 times does
   more damage than one never cited. Rows are banded by real corpus frequency
   from CITATION_STRINGS so the sample reflects impact, not just row count.

Sampling is seeded, so the draw is reproducible and can be re-derived by a
reviewer who doubts it.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.schema import RegistryDB

OUT = REPO / "registry" / "out"
SEED = 20260905
# Items appearing in BOTH samples, so the two reviewers judge some of the same
# rows independently and an agreement statistic can be computed.
OVERLAP = 30
SINGLE_SOURCED = ("ipc-1860", "crpc-1973", "iea-1872")

HEADERS = [
    "sample_id", "stratum", "priority",
    "from_id", "from_heading", "relation", "effective_on",
    "to_id", "to_heading",
    "source_doc", "corpus_citations",
    "VERDICT (correct / wrong / unclear)", "CORRECTED_VALUE", "NOTES", "CHECKED_BY", "DATE",
]


def fetch_rows(db):
    """Every supersession mapping, annotated with source strength and frequency."""
    return db.db.execute("""
        SELECT
            sup.from_section_id,
            COALESCE(fs.heading, '')                       AS from_heading,
            sup.relation,
            CAST(sup.effective_on AS VARCHAR)              AS effective_on,
            COALESCE(sup.to_section_id, '')                AS to_id,
            COALESCE(ts.heading, '')                       AS to_heading,
            COALESCE(sup.source_doc, '')                   AS source_doc,
            sup.from_statute_id,
            COALESCE((SELECT sum(cs.occurrences) FROM CITATION_STRINGS cs
                      WHERE cs.target_id = sup.from_section_id), 0) AS cites,
            (SELECT count(*) FROM SUPERSESSION s2
             WHERE s2.from_section_id = sup.from_section_id)        AS fanout
        FROM SUPERSESSION sup
        LEFT JOIN SECTIONS fs ON fs.id = sup.from_section_id
        LEFT JOIN SECTIONS ts ON ts.id = sup.to_section_id
        ORDER BY sup.from_section_id, sup.to_section_id
    """).fetchall()


def stratum_of(r) -> str:
    (_fid, _fh, relation, _eff, _tid, _th, _src, statute, cites, fanout) = r
    src = "single-source" if statute in SINGLE_SOURCED else "cross-validated"
    if relation == "repealed_no_successor":
        kind = "no-successor"
    elif fanout > 1:
        kind = "one-to-many"
    else:
        kind = "one-to-one"
    if cites >= 50:
        band = "high-cited"
    elif cites >= 5:
        band = "mid-cited"
    elif cites > 0:
        band = "low-cited"
    else:
        band = "uncited"
    return f"{src} | {kind} | {band}"


def priority_of(stratum: str) -> str:
    if "single-source" in stratum and "high-cited" in stratum:
        return "1-critical"
    if "single-source" in stratum:
        return "2-high"
    if "high-cited" in stratum or "no-successor" in stratum or "one-to-many" in stratum:
        return "3-medium"
    return "4-routine"


# Share of each sample to spend on each priority band. Proportional-to-size
# allocation would spend ~90% of the sample on routine 1:1 mappings, because
# that is what most rows are -- and would put ~2 rows on the highest-impact
# stratum. The point of stratifying is to buy precision where an error costs
# most, so the bands get fixed budgets instead.
PRIORITY_SHARE = {
    "1-critical": 0.25,
    "2-high": 0.45,
    "3-medium": 0.20,
    "4-routine": 0.10,
}


def draw(buckets: dict[str, list], n: int, rng, taken: set) -> list:
    """Priority-budgeted allocation, with a floor of 1 per non-empty stratum."""
    by_priority: dict[str, list[str]] = {}
    for k in buckets:
        by_priority.setdefault(priority_of(k), []).append(k)

    picked: list = []

    def take(stratum_keys, budget):
        got = 0
        # floor: one from each stratum in this band
        for k in stratum_keys:
            pool = [x for x in buckets[k] if id(x) not in taken]
            if pool and got < budget and len(picked) < n:
                c = rng.choice(pool)
                taken.add(id(c))
                picked.append((k, c))
                got += 1
        # fill the rest of the band uniformly across its strata
        while got < budget and len(picked) < n:
            pool_all = [(k, x) for k in stratum_keys
                        for x in buckets[k] if id(x) not in taken]
            if not pool_all:
                break
            k, c = rng.choice(pool_all)
            taken.add(id(c))
            picked.append((k, c))
            got += 1

    for band in ("1-critical", "2-high", "3-medium", "4-routine"):
        keys = by_priority.get(band, [])
        if keys:
            take(keys, round(n * PRIORITY_SHARE[band]))

    # Any shortfall (a band ran out of rows) is redistributed to the highest
    # remaining priority rather than left unsampled.
    for band in ("1-critical", "2-high", "3-medium", "4-routine"):
        if len(picked) >= n:
            break
        keys = by_priority.get(band, [])
        if keys:
            take(keys, n - len(picked))

    return picked


def write_sheet(path: Path, picked: list, prefix: str) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(HEADERS)
        for i, (stratum, r) in enumerate(sorted(picked, key=lambda p: (priority_of(p[0]), p[0])), 1):
            (fid, fh_, relation, eff, tid, th, src, _st, cites, _fan) = r
            w.writerow([
                "%s-%03d" % (prefix, i), stratum, priority_of(stratum),
                fid, fh_, relation, eff or "", tid, th, src, cites,
                "", "", "", "", "",
            ])


def main() -> int:
    db = RegistryDB()
    rows = fetch_rows(db)
    print("supersession mappings available: %d" % len(rows))

    buckets: dict[str, list] = {}
    for r in rows:
        buckets.setdefault(stratum_of(r), []).append(r)

    print("\nstrata (%d):" % len(buckets))
    for k in sorted(buckets, key=lambda s: (priority_of(s), s)):
        print("  %-11s %-46s %d rows" % (priority_of(k), k, len(buckets[k])))

    rng = random.Random(SEED)
    taken: set = set()

    # The independent audit is drawn FIRST. It is the higher-stakes sample
    # (threat T10) and the scarce high-impact strata should go to it rather
    # than be consumed by the self-check.
    s200 = draw(buckets, 200, rng, taken)
    s100_fresh = draw(buckets, 100 - OVERLAP, rng, taken)

    # Deliberate overlap. RESEARCH.md 7.5 requires an agreement statistic for
    # every study, and Cohen's kappa needs the two people to have judged the
    # SAME items independently. Fully disjoint samples make agreement
    # uncomputable, so a shared block is built in on purpose.
    #
    # The shared rows are taken FROM the audit sample and added to the
    # self-check -- not the reverse, which would duplicate rows inside one
    # sheet instead of sharing them across two.
    shared = rng.sample(s200, min(OVERLAP, len(s200)))
    s100 = s100_fresh + shared

    OUT.mkdir(parents=True, exist_ok=True)
    write_sheet(OUT / "verify_100.csv", s100, "V")
    write_sheet(OUT / "audit_200.csv", s200, "A")

    def mix_of(sample):
        m: dict[str, int] = {}
        for stratum, _ in sample:
            m[priority_of(stratum)] = m.get(priority_of(stratum), 0) + 1
        return dict(sorted(m.items()))

    print("\nverify_100.csv : %d rows   %s" % (len(s100), mix_of(s100)))
    print("audit_200.csv  : %d rows   %s" % (len(s200), mix_of(s200)))
    print("shared items   : %d (for inter-rater agreement, RESEARCH.md 7.5)" % len(shared))
    print("\nseed = %d, so this draw is reproducible and re-checkable." % SEED)
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
