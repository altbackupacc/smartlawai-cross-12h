"""M1 T4: score completed human-validation rater CSVs (from
sample_human_validation.py) against HHEM's own accept/reject verdicts, and
(with 2+ raters) inter-rater agreement -- per RESEARCH.md sec7.5's
"report agreement for every study" mandate.

Binarization for HHEM-agreement purposes: a rater's "Accept" counts as
agreeing with HHEM's accept; "Revise" and "Reject" both count as agreeing
with HHEM's reject (a section a human would edit is not one that should
enter training data unedited, same as an outright rejection). Cohen's kappa
across raters is computed on the original 3-way verdicts, not this
collapsed version -- collapsing loses information agreement itself should
capture.

Usage:
    python data/score_human_validation.py --raters data/human_validation/rater1_sample.csv
    python data/score_human_validation.py --raters data/human_validation/rater1_sample.csv data/human_validation/rater2_sample.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ANSWER_KEY_PATH = Path("data/human_validation/answer_key.json")
VALID_VERDICTS = {"accept", "revise", "reject"}


def load_rater_csv(path: Path) -> dict[str, str]:
    verdicts = {}
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        v = row["rater_verdict"].strip().lower()
        if not v:
            raise SystemExit(f"{path}: item_id {row['item_id']} has no rater_verdict -- "
                              f"all {len(rows)} rows must be filled in before scoring.")
        if v not in VALID_VERDICTS:
            raise SystemExit(f"{path}: item_id {row['item_id']} has invalid verdict "
                              f"'{row['rater_verdict']}' -- must be one of "
                              f"Accept/Revise/Reject.")
        verdicts[row["item_id"]] = v
    return verdicts


def cohens_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    categories = sorted(set(a) | set(b))
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in categories)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def score_against_hhem(verdicts: dict[str, str], answer_key: dict[str, dict]) -> dict:
    agree, disagree = 0, 0
    confusion: Counter[tuple[str, str]] = Counter()
    for item_id, verdict in verdicts.items():
        hhem_accepted = answer_key[item_id]["hhem_accepted"]
        human_accepts = verdict == "accept"
        confusion[(hhem_accepted, human_accepts)] += 1
        if hhem_accepted == human_accepts:
            agree += 1
        else:
            disagree += 1
    total = agree + disagree
    return {
        "n_items": total,
        "agreement_rate_pct": round(agree / total * 100, 2),
        "confusion": {
            "hhem_accept_human_accept": confusion[(True, True)],
            "hhem_accept_human_reject": confusion[(True, False)],  # false accept
            "hhem_reject_human_accept": confusion[(False, True)],  # false reject
            "hhem_reject_human_reject": confusion[(False, False)],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raters", nargs="+", required=True, type=Path,
                         help="One or more completed rater{N}_sample.csv files.")
    parser.add_argument("--answer-key", type=Path, default=ANSWER_KEY_PATH)
    args = parser.parse_args()

    answer_key = json.loads(args.answer_key.read_text(encoding="utf-8"))
    rater_verdicts = [load_rater_csv(p) for p in args.raters]

    for path, verdicts in zip(args.raters, rater_verdicts):
        missing = set(answer_key) - set(verdicts)
        if missing:
            raise SystemExit(f"{path} is missing verdicts for {len(missing)} items: "
                              f"{sorted(missing)[:5]}...")
        result = score_against_hhem(verdicts, answer_key)
        print(f"\n=== {path.name} vs HHEM ===")
        print(json.dumps(result, indent=2))

    if len(rater_verdicts) >= 2:
        print("\n=== Inter-rater agreement (Cohen's kappa, pairwise, 3-way verdicts) ===")
        item_ids = sorted(set(rater_verdicts[0]))
        for i in range(len(rater_verdicts)):
            for j in range(i + 1, len(rater_verdicts)):
                a = [rater_verdicts[i][k] for k in item_ids]
                b = [rater_verdicts[j][k] for k in item_ids]
                kappa = cohens_kappa(a, b)
                print(f"  {args.raters[i].name} vs {args.raters[j].name}: "
                      f"kappa = {kappa:.4f}")

    print("\nReport these numbers plainly in data/PROVENANCE.md (RESEARCH.md T4), "
          "whatever they turn out to be.")


if __name__ == "__main__":
    main()
