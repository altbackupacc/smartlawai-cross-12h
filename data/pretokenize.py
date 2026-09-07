"""M1 step 9: precompute InLegalBERT token ids for the frozen section-pair
corpus, so M2 (retrieval), M5 (verification), and M8a (encoder fine-tuning)
don't each redundantly re-tokenize the same text with the same tokenizer
(see RESEARCH.md sec2.5). Does NOT touch M8b's QLoRA tokenizers -- those are
model-specific (Qwen2.5-3B/7B/14B, Mistral-7B) and stay entirely M8b's job.

Must run AFTER dedup_split.py, so the `split` column is final and frozen
before tokenization happens (I5).

Input:  data/processed/section_pairs.jsonl
Output: data/processed/section_pairs.arrow/ (datasets.Dataset.save_to_disk())
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

IN_PATH = Path("data/processed/section_pairs.jsonl")
OUT_PATH = Path("data/processed/section_pairs.arrow")
TOKENIZER_ID = "law-ai/InLegalBERT"  # matches src/smartlawai/core/inlegalbert.py's MODEL_ID
MAX_LEN = 512  # matches core/inlegalbert.py's default


def load_pairs(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(pairs: list[dict], tokenizer) -> Dataset:
    doc_ids = [p["doc_id"] for p in pairs]
    splits = [p["split"] for p in pairs]
    section_ids = [p["section"] for p in pairs]
    source_texts = [p["source_headnote"] for p in pairs]
    target_texts = [p["generated_text"] for p in pairs]
    provenances = [p["provenance"] for p in pairs]

    source_enc = tokenizer(source_texts, max_length=MAX_LEN, truncation=True,
                            padding="max_length")
    target_enc = tokenizer(target_texts, max_length=MAX_LEN, truncation=True,
                            padding="max_length")

    return Dataset.from_dict({
        "doc_id": doc_ids,
        "split": splits,
        "section_id": section_ids,
        "source_text": source_texts,
        "target_text": target_texts,
        "provenance": provenances,
        "source_input_ids": source_enc["input_ids"],
        "source_attention_mask": source_enc["attention_mask"],
        "target_input_ids": target_enc["input_ids"],
        "target_attention_mask": target_enc["attention_mask"],
    })


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"{IN_PATH} not found -- run dedup_split.py first (I5: "
                          f"tokenization must happen after the split is frozen).")

    print(f"Loading {IN_PATH}...")
    pairs = load_pairs(IN_PATH)
    print(f"  {len(pairs)} section pairs loaded.")

    print(f"Loading tokenizer {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    print(f"Tokenizing (max_length={MAX_LEN})...")
    ds = build_dataset(pairs, tokenizer)

    print(f"Saving to {OUT_PATH}...")
    ds.save_to_disk(str(OUT_PATH))

    by_split: dict[str, int] = {}
    for s in ds["split"]:
        by_split[s] = by_split.get(s, 0) + 1
    print(f"\n=== DONE === {len(ds)} rows -> {OUT_PATH}")
    print(f"  by split: {by_split}")
    print(f"  columns: {ds.column_names}")


if __name__ == "__main__":
    main()
