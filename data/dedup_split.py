"""M1 step 6+7: MinHash near-duplicate detection + frozen document-level
split, then join the HHEM-accepted section pairs against that split to
produce the final training corpus.

Per M1_ONBOARDING.md #7 and I5 (CLAUDE.md): document-level splits are
computed ONCE and recorded; this script refuses to overwrite an existing
frozen split without --force, and --force alone does not excuse writing the
required manual addendum to data/PROVENANCE.md explaining why the freeze
was broken.

Dedup is computed over summ's full judgment text (not the generated section
pairs) so that near-duplicate *source documents* never end up split across
train/dev/test -- that's the actual leakage risk I5 guards against.

Usage:
    python data/dedup_split.py
    python data/dedup_split.py --force   # only after adding a PROVENANCE.md
                                          # addendum explaining why
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from datasets import load_from_disk
from datasketch import MinHash, MinHashLSH

RAW_SUMM_DIR = Path("data/raw/summ")
ACCEPTED_PAIRS_PATH = Path("data/processed/section_pairs_accepted.jsonl")
SPLITS_DIR = Path("data/splits")
NEAR_DUP_PATH = Path("data/processed/near_duplicate_pairs.json")
FINAL_PAIRS_PATH = Path("data/processed/section_pairs.jsonl")

SHINGLE_SIZE = 5        # word 5-grams, per PLAN.md
NUM_PERM = 128
LSH_THRESHOLD = 0.9     # PLAN.md's literal spec -- don't change without updating that file
SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
SPLIT_SEED = 42

# Boilerplate lines that precede almost every judgment (case numbers, bench
# composition, counsel names, filing dates) generate spurious shingle overlap
# between otherwise-unrelated judgments. Stripped before shingling so MinHash
# similarity reflects the actual judgment text, not shared header formatting.
_HEADER_LINE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^(civil|criminal|writ)\s+(appeal|petition)\b",
        r"^i?vil\s+appeal\b",
        r"^from\s+the\s+judgment",
        r"^appeal\s+no\.?",
        r"\bfor\s+the\s+(appellant|respondent)s?\b",
        r"^\s*j\.?\s*$",
        r"\bair\s+\d{4}\b",
        r"\bscr\s+\d+\b",
        r"^\d{1,2}\.\d{1,2}\.\d{2,4}",
    ]
]


def strip_header_noise(paragraphs: list[str]) -> list[str]:
    return [p for p in paragraphs
            if not any(pat.search(p) for pat in _HEADER_LINE_PATTERNS)]


def normalize_text(paragraphs: list[str]) -> str:
    text = " ".join(strip_header_noise(paragraphs))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def build_minhash(shingle_set: set[str]) -> MinHash:
    # update_batch (vectorized) instead of a per-item .update() loop -- the
    # per-item path's fixed per-call overhead dominates at this corpus's
    # scale (~28.5M total shingles across 7,130 docs, one doc alone has
    # 117k), turning what should be a ~1-minute job into 60+ minutes.
    mh = MinHash(num_perm=NUM_PERM)
    if shingle_set:
        mh.update_batch([s.encode("utf-8") for s in shingle_set])
    return mh


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_near_duplicates(doc_texts: dict[str, str]) -> tuple[list[dict], dict[str, list[str]]]:
    """Returns (pairwise edges with estimated jaccard, doc_id -> shingle-set-derived
    minhash signature not needed downstream) and builds connected components."""
    minhashes: dict[str, MinHash] = {}
    shingle_sets: dict[str, set[str]] = {}
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)

    for doc_id, text in doc_texts.items():
        sh = shingles(text)
        shingle_sets[doc_id] = sh
        mh = build_minhash(sh)
        minhashes[doc_id] = mh
        lsh.insert(doc_id, mh)

    uf = UnionFind(list(doc_texts.keys()))
    edges = []
    seen_pairs = set()
    for doc_id, mh in minhashes.items():
        for neighbor in lsh.query(mh):
            if neighbor == doc_id:
                continue
            pair_key = tuple(sorted((doc_id, neighbor)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            est_jaccard = minhashes[pair_key[0]].jaccard(minhashes[pair_key[1]])
            if est_jaccard >= LSH_THRESHOLD:
                edges.append({"doc_a": pair_key[0], "doc_b": pair_key[1],
                              "estimated_jaccard": round(float(est_jaccard), 4)})
                uf.union(pair_key[0], pair_key[1])

    components: dict[str, list[str]] = defaultdict(list)
    for doc_id in doc_texts:
        components[uf.find(doc_id)].append(doc_id)
    return edges, {root: members for root, members in components.items() if len(members) > 1}


def resolve_components(components: dict[str, list[str]],
                        doc_texts: dict[str, str]) -> tuple[set[str], list[dict]]:
    """For each connected component of near-duplicates, keep the longest text
    (ties broken by lowest numeric doc_id). Returns dropped doc_ids and a
    per-component record of what was kept/dropped."""
    dropped: set[str] = set()
    component_records = []
    for members in components.values():
        ranked = sorted(members, key=lambda d: (-len(doc_texts[d]), int(d)))
        kept, rest = ranked[0], ranked[1:]
        dropped.update(rest)
        component_records.append({"members": sorted(members, key=int),
                                   "kept": kept, "dropped": sorted(rest, key=int)})
    return dropped, component_records


def make_splits(surviving_doc_ids: list[str]) -> dict[str, list[str]]:
    ids = sorted(surviving_doc_ids, key=int)
    random.Random(SPLIT_SEED).shuffle(ids)
    n = len(ids)
    n_train = round(n * SPLIT_RATIOS["train"])
    n_dev = round(n * SPLIT_RATIOS["dev"])
    return {
        "train": ids[:n_train],
        "dev": ids[n_train:n_train + n_dev],
        "test": ids[n_train + n_dev:],
    }


def write_splits(splits: dict[str, list[str]]) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    frozen_at = datetime.now(UTC).date().isoformat()
    for split_name, doc_ids in splits.items():
        payload = {
            "split": split_name,
            "source": "IL-TUR/summ",
            "doc_ids": doc_ids,
            "count": len(doc_ids),
            "dedup_threshold": LSH_THRESHOLD,
            "seed": SPLIT_SEED,
            "frozen_at": frozen_at,
        }
        (SPLITS_DIR / f"{split_name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")


def join_pairs(doc_to_split: dict[str, str], dropped_doc_ids: set[str]) -> dict[str, int]:
    """Join the HHEM-accepted section pairs against the frozen split,
    dropping any pair whose source document was removed as a near-duplicate,
    and overwriting the pre-existing `split` field (IL-TUR's own train/test
    split, unrelated to our frozen doc-level split) with our own."""
    counts = {"train": 0, "dev": 0, "test": 0, "dropped_as_duplicate": 0}
    with open(ACCEPTED_PAIRS_PATH, encoding="utf-8") as f_in, \
         open(FINAL_PAIRS_PATH, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            pair = json.loads(line)
            doc_id = pair["doc_id"]
            if doc_id in dropped_doc_ids:
                counts["dropped_as_duplicate"] += 1
                continue
            split = doc_to_split.get(doc_id)
            if split is None:
                raise ValueError(f"doc_id {doc_id} has no split assignment -- "
                                  f"not in summ's id set?")
            pair["iltur_original_split"] = pair.get("split")
            pair["split"] = split
            f_out.write(json.dumps(pair) + "\n")
            counts[split] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing frozen split. Per I5, this alone "
                              "does NOT excuse skipping a manual data/PROVENANCE.md "
                              "addendum explaining why the freeze was broken -- add "
                              "that yourself before running with this flag.")
    args = parser.parse_args()

    existing = [p for p in ("train", "dev", "test") if (SPLITS_DIR / f"{p}.json").exists()]
    if existing and not args.force:
        raise SystemExit(
            f"Refusing to overwrite existing frozen split(s) {existing} in {SPLITS_DIR}/ "
            f"(I5: document-level splits are recorded, never regenerated). "
            f"If this is a deliberate, documented exception, first add a manual "
            f"addendum to data/PROVENANCE.md explaining why, then re-run with --force."
        )
    if existing and args.force:
        print(f"--force passed: overwriting {existing}. Make sure data/PROVENANCE.md "
              f"already documents why this freeze was broken.")

    print("Loading summ (full judgment text)...")
    summ = load_from_disk(str(RAW_SUMM_DIR))
    doc_texts: dict[str, str] = {}
    for split_name in summ:
        for row in summ[split_name]:
            doc_texts[row["id"]] = normalize_text(row["document"])
    print(f"  {len(doc_texts)} unique documents loaded.")

    print(f"Computing MinHash (num_perm={NUM_PERM}) + LSH (threshold={LSH_THRESHOLD})...")
    edges, components = find_near_duplicates(doc_texts)
    dropped_doc_ids, component_records = resolve_components(components, doc_texts)
    print(f"  {len(edges)} near-duplicate pairs found across {len(components)} "
          f"connected components; dropping {len(dropped_doc_ids)} documents.")

    NEAR_DUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEAR_DUP_PATH.write_text(json.dumps({
        "shingle_size": SHINGLE_SIZE,
        "num_perm": NUM_PERM,
        "lsh_threshold": LSH_THRESHOLD,
        "n_docs_checked": len(doc_texts),
        "n_near_duplicate_docs_dropped": len(dropped_doc_ids),
        "pairs": edges,
        "components": component_records,
    }, indent=2), encoding="utf-8")
    print(f"  wrote {NEAR_DUP_PATH}")

    surviving = [d for d in doc_texts if d not in dropped_doc_ids]
    splits = make_splits(surviving)
    write_splits(splits)
    print(f"  split sizes: train={len(splits['train'])}, dev={len(splits['dev'])}, "
          f"test={len(splits['test'])} (of {len(surviving)} surviving docs)")

    doc_to_split = {d: s for s, ids in splits.items() for d in ids}
    print(f"Joining {ACCEPTED_PAIRS_PATH} against the frozen split...")
    counts = join_pairs(doc_to_split, dropped_doc_ids)
    print(f"  wrote {FINAL_PAIRS_PATH}: train={counts['train']}, dev={counts['dev']}, "
          f"test={counts['test']}, dropped_as_duplicate={counts['dropped_as_duplicate']}")

    print("\n=== FREEZE POINT REACHED ===")
    print("data/splits/{train,dev,test}.json are now frozen per I5.")
    print("Next: update data/PROVENANCE.md with these counts, then commit "
          "data/splits/, data/processed/near_duplicate_pairs.json, and this script.")


if __name__ == "__main__":
    main()
