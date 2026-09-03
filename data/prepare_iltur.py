"""Pull + cache IL-TUR configs from HuggingFace (M1 step 1, PLAN.md).
Idempotent: skips a config whose target dir already holds a valid DatasetDict.
No filtering, no cleaning -- that happens in align_rr_summ.py/build_pairs.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASET_ID = "Exploration-Lab/IL-TUR"
CONFIGS = ["summ", "rr", "lner", "lsi"]
RAW_DIR = Path("data/raw")


def _is_valid_dataset_dir(path: Path) -> bool:
    return path.exists() and (path / "dataset_dict.json").exists()


def pull_config(config_name: str, cache_dir: Path = RAW_DIR, force: bool = False) -> Path:
    out_dir = cache_dir / config_name
    if _is_valid_dataset_dir(out_dir) and not force:
        print(f"[{config_name}] already cached at {out_dir}, skipping (use --force to re-pull)")
        return out_dir

    from datasets import load_dataset

    print(f"[{config_name}] downloading from {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, config_name)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"[{config_name}] saved to {out_dir}")
    for split_name, split in ds.items():
        print(f"  {split_name}: {len(split)} rows, features={list(split.features.keys())}")
    return out_dir


def record_revision(out_path: Path = RAW_DIR / "REVISION.json") -> None:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET_ID)
    payload = {"dataset_id": DATASET_ID, "sha": info.sha, "last_modified": str(info.last_modified)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Recorded dataset revision to {out_path}: {payload}")


def main(configs: list[str], force: bool = False) -> None:
    for name in configs:
        pull_config(name, force=force)
    record_revision()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", dest="configs", choices=CONFIGS,
                         help="Pull only this config (repeatable). Default: all four.")
    parser.add_argument("--force", action="store_true", help="Re-pull even if cached.")
    args = parser.parse_args()
    main(args.configs or CONFIGS, force=args.force)
