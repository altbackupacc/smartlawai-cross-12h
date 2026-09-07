"""M1 T4: sample + export the 200-item human validation set (RESEARCH.md T4 --
"human-validate 200, report acceptance rate"). This checks whether HHEM's own
accept/reject verdict agrees with a human reader's, on BOTH accepted and
rejected pairs -- not just the accepted ones, otherwise there's no way to
measure how many genuinely-bad pairs the filter let through, or how many
genuinely-fine pairs it wrongly rejected.

Design: 100 accepted + 100 rejected (stratified, not proportional to the
corpus's real 96.5%/3.5% split) -- proportional sampling would give ~193
accepted / ~7 rejected, which is far too few rejected items to say anything
about the filter's false-accept vs false-reject behavior. This is a
diagnostic sample of the filter, not a population estimate of the corpus.

Excludes any pair whose source document was dropped as a near-duplicate
(data/processed/near_duplicate_pairs.json) -- those documents never made it
into the final corpus, so validating pairs from them would validate content
nobody will ever train on.

Output:
  data/human_validation/rater{N}_sample.csv  -- one per --num-raters, blind
      (no HHEM verdict shown), identical content, for independent annotation.
  data/human_validation/answer_key.json      -- item_id -> HHEM's own verdict,
      kept separately so raters never see it while annotating.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

SCORED_PATH = Path("data/processed/section_pairs_scored.jsonl")
NEAR_DUP_PATH = Path("data/processed/near_duplicate_pairs.json")
OUT_DIR = Path("data/human_validation")
N_ACCEPTED = 100
N_REJECTED = 100
SEED = 42


def dropped_doc_ids() -> set[str]:
    if not NEAR_DUP_PATH.exists():
        return set()
    data = json.loads(NEAR_DUP_PATH.read_text(encoding="utf-8"))
    return {d for comp in data["components"] for d in comp["dropped"]}


def load_pool() -> list[dict]:
    dropped = dropped_doc_ids()
    pairs = []
    with open(SCORED_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["doc_id"] in dropped:
                continue
            pairs.append(row)
    return pairs


def sample_items(pairs: list[dict], n_accepted: int, n_rejected: int,
                  seed: int) -> list[dict]:
    accepted = [p for p in pairs if p["hhem_accepted"]]
    rejected = [p for p in pairs if not p["hhem_accepted"]]
    rng = random.Random(seed)
    if len(accepted) < n_accepted or len(rejected) < n_rejected:
        raise SystemExit(
            f"Not enough pairs to sample: {len(accepted)} accepted (need "
            f"{n_accepted}), {len(rejected)} rejected (need {n_rejected})."
        )
    sampled = rng.sample(accepted, n_accepted) + rng.sample(rejected, n_rejected)
    rng.shuffle(sampled)  # interleave accept/reject so raters can't tell which is which
    for row in sampled:
        row["item_id"] = f"{row['doc_id']}__{row['section']}"
    return sampled


def write_rater_csv(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["item_id", "doc_id", "section", "source_headnote",
                          "generated_text", "rater_verdict", "rater_notes"])
        for row in items:
            writer.writerow([row["item_id"], row["doc_id"], row["section"],
                              row["source_headnote"], row["generated_text"],
                              "", ""])


def write_answer_key(items: list[dict], path: Path) -> None:
    key = {row["item_id"]: {"hhem_accepted": row["hhem_accepted"],
                             "hhem_pass_rate": row["hhem_pass_rate"],
                             "hhem_mean_score": row["hhem_mean_score"]}
           for row in items}
    path.write_text(json.dumps(key, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-raters", type=int, default=1,
                         help="Number of identical, independent rater CSVs to write.")
    parser.add_argument("--n-accepted", type=int, default=N_ACCEPTED)
    parser.add_argument("--n-rejected", type=int, default=N_REJECTED)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"Loading {SCORED_PATH}...")
    pairs = load_pool()
    print(f"  {len(pairs)} pairs after excluding near-duplicate-dropped documents.")

    items = sample_items(pairs, args.n_accepted, args.n_rejected, args.seed)
    print(f"  sampled {len(items)} items ({args.n_accepted} accepted + "
          f"{args.n_rejected} rejected, interleaved).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, args.num_raters + 1):
        out_path = OUT_DIR / f"rater{i}_sample.csv"
        write_rater_csv(items, out_path)
        print(f"  wrote {out_path}")

    answer_key_path = OUT_DIR / "answer_key.json"
    write_answer_key(items, answer_key_path)
    print(f"  wrote {answer_key_path} (NOT for raters -- HHEM verdicts, kept for scoring)")

    print("\nNext: have each rater fill in the 'rater_verdict' column "
          "(Accept / Revise / Reject) in their own rater{N}_sample.csv, "
          "then run data/score_human_validation.py.")


if __name__ == "__main__":
    main()
