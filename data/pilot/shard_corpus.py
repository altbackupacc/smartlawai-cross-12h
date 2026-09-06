"""Split section_pairs_raw.jsonl into weighted shards for parallel GPU scoring
(4 L4 + 1 A100, A100 weighted ~2.5x an L4 based on typical throughput -- an
estimate, not a measurement, but reasonable to start with)."""
import json
from pathlib import Path

IN_PATH = Path("data/processed/section_pairs_raw.jsonl")
OUT_DIR = Path("data/processed/shards")
WEIGHTS = {"l4_1": 1, "l4_2": 1, "l4_3": 1, "l4_4": 1, "a100": 2.5}

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(IN_PATH, encoding="utf-8") as f:
    lines = f.readlines()

total = len(lines)
total_weight = sum(WEIGHTS.values())
idx = 0
for name, w in WEIGHTS.items():
    n = round(total * (w / total_weight))
    shard_lines = lines[idx:idx + n]
    idx += n
    out_path = OUT_DIR / f"section_pairs_{name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(shard_lines)
    print(f"{name}: {len(shard_lines)} pairs -> {out_path}")

# leftover (rounding) goes to the last shard
if idx < total:
    leftover = lines[idx:]
    last_name = list(WEIGHTS.keys())[-1]
    out_path = OUT_DIR / f"section_pairs_{last_name}.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.writelines(leftover)
    print(f"appended {len(leftover)} leftover pairs to {out_path}")
