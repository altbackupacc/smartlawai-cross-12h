"""Score the filled verification worksheets.

Produces the two numbers PLAN.md M4 and RESEARCH.md 7.5 require:

* **verification accuracy** — the share of sampled mappings a human confirmed
  correct. This is what goes in the paper as "parse verification accuracy",
  and it is the answer to threat T10.
* **inter-rater agreement (Cohen's kappa)** — computed on the 30 rows both
  reviewers judged independently. RESEARCH.md 7.5: report agreement for every
  study. An accuracy figure with no agreement figure behind it is one person's
  opinion.

Run after both CSVs have their VERDICT column filled. Rows left blank are
reported as unreviewed rather than counted as correct.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "registry" / "out"

VERDICT = "VERDICT (correct / wrong / unclear)"


def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"missing: {path}", file=sys.stderr)
        return []
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def norm(v: str | None) -> str | None:
    if not v:
        return None
    v = v.strip().lower()
    for k in ("correct", "wrong", "unclear"):
        if v.startswith(k):
            return k
    return None


def kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two raters over the same items."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def report(name: str, rows: list[dict]) -> dict[str, str]:
    verdicts = {r["from_id"] + "|" + r["to_id"]: norm(r.get(VERDICT)) for r in rows}
    done = {k: v for k, v in verdicts.items() if v}
    c = Counter(done.values())
    total, n = len(rows), len(done)

    print(f"\n=== {name} ===")
    print("  rows: %d   reviewed: %d   unreviewed: %d" % (total, n, total - n))
    if not n:
        print("  (nothing scored yet)")
        return verdicts

    scoreable = c["correct"] + c["wrong"]
    acc = 100 * c["correct"] / scoreable if scoreable else 0.0
    print("  correct: %d   wrong: %d   unclear: %d" % (c["correct"], c["wrong"], c["unclear"]))
    print("  VERIFICATION ACCURACY: %.1f%%  (%d/%d, excluding 'unclear')"
          % (acc, c["correct"], scoreable))
    if n < total:
        print("  NOTE: %d rows unreviewed -- accuracy is provisional" % (total - n))

    by = {}
    for r in rows:
        v = norm(r.get(VERDICT))
        if not v:
            continue
        by.setdefault(r["priority"], Counter())[v] += 1
    if by:
        print("  by priority:")
        for k in sorted(by):
            cc = by[k]
            s = cc["correct"] + cc["wrong"]
            print("    %-11s %d correct / %d scored%s"
                  % (k, cc["correct"], s, "  (%.0f%%)" % (100 * cc["correct"] / s) if s else ""))
    return verdicts


def main() -> int:
    v = load(OUT / "verify_100.csv")
    a = load(OUT / "audit_200.csv")
    if not v and not a:
        return 1

    vv = report("verify_100 (self-check vs Gazette)", v)
    av = report("audit_200 (independent audit, threat T10)", a)

    shared = sorted(set(vv) & set(av))
    both = [(vv[k], av[k]) for k in shared if vv[k] and av[k]]
    print("\n=== inter-rater agreement (RESEARCH.md 7.5) ===")
    print("  shared items: %d   judged by both: %d" % (len(shared), len(both)))
    if both:
        k = kappa([x for x, _ in both], [y for _, y in both])
        agree = 100 * sum(1 for x, y in both if x == y) / len(both)
        print(f"  raw agreement: {agree:.1f}%")
        print(f"  Cohen's kappa: {k:.3f}")
        band = ("poor" if k < 0.2 else "fair" if k < 0.4 else "moderate"
                if k < 0.6 else "substantial" if k < 0.8 else "almost perfect")
        print(f"  interpretation: {band}")
        dis = [(kk, vv[kk], av[kk]) for kk in shared
               if vv[kk] and av[kk] and vv[kk] != av[kk]]
        if dis:
            print("  disagreements (resolve these by discussion, do not average):")
            for kk, x, y in dis[:12]:
                print("    %-40s self=%-8s audit=%s" % (kk, x, y))
    else:
        print("  (not computable until both sheets have the shared rows filled)")

    print("\nReport BOTH numbers in the paper: accuracy alone, without an")
    print("agreement figure, is a single reviewer's opinion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
