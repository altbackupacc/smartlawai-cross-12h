"""M1 leakage tests (I5). Verifies the frozen splits in data/splits/*.json
never share a document id, that no recorded near-duplicate pair spans two
splits, and (once M6 lands) that no eval item derives from a training doc."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

SPLITS_DIR = Path("data/splits")
NEAR_DUP_PATH = Path("data/processed/near_duplicate_pairs.json")


def _load_splits() -> dict[str, set[str]]:
    splits = {}
    for name in ("train", "dev", "test"):
        path = SPLITS_DIR / f"{name}.json"
        if not path.exists():
            pytest.skip(f"{path} not found -- dedup_split.py hasn't run yet")
        splits[name] = set(json.loads(path.read_text(encoding="utf-8"))["doc_ids"])
    return splits


def test_no_doc_id_in_two_splits():
    splits = _load_splits()
    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = splits[names[i]] & splits[names[j]]
            assert not overlap, (
                f"doc_id(s) {overlap} appear in both '{names[i]}' and '{names[j]}' splits"
            )


def test_no_near_duplicate_spans_splits():
    splits = _load_splits()
    if not NEAR_DUP_PATH.exists():
        pytest.skip(f"{NEAR_DUP_PATH} not found -- dedup_split.py hasn't run yet")
    doc_to_split = {doc_id: name for name, ids in splits.items() for doc_id in ids}
    pairs = json.loads(NEAR_DUP_PATH.read_text(encoding="utf-8"))["pairs"]
    for pair in pairs:
        a, b = pair["doc_a"], pair["doc_b"]
        split_a, split_b = doc_to_split.get(a), doc_to_split.get(b)
        if split_a is None or split_b is None:
            # one or both members were dropped as a duplicate and never
            # entered any split -- nothing to check for that member.
            continue
        assert split_a == split_b, (
            f"near-duplicate pair ({a}, {b}) spans splits '{split_a}' and '{split_b}'"
        )


def assert_no_eval_leakage(eval_glob: str, doc_id_field: str, train_split: str) -> None:
    eval_files = glob.glob(eval_glob)
    if not eval_files:
        pytest.skip(f"no files matching {eval_glob} yet -- M6 not landed")
    train_doc_ids = set(json.loads(Path(train_split).read_text(encoding="utf-8"))["doc_ids"])
    for path in eval_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                doc_id = item.get(doc_id_field)
                assert doc_id not in train_doc_ids, (
                    f"eval item in {path} derives from training doc_id {doc_id}"
                )


def test_no_qa_eval_item_from_training_doc():
    assert_no_eval_leakage(
        eval_glob="eval/gold/*.jsonl",
        doc_id_field="source_doc_id",
        train_split="data/splits/train.json",
    )
