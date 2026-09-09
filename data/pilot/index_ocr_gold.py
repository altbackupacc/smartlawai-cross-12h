"""Index the 50-page OCR-WER gold set (data/ocr_wer_gold/) into short
sequential filenames for easier manual review, and convert filled-in human
transcripts back into the page_id-keyed layout eval/ocr_wer.py expects.

The real gold set is keyed by page_id (e.g.
KLHC010000011948_1_1958-03-03_p002) because that's what ties a page back to
its source document/page number. This script never renames the underlying
data -- it only adds a parallel, human-friendlier view:

    export: ocr_model_{i}.txt   <- GLM-OCR prediction for the i-th page
            human_ocr_{i}.txt   <- blank starter file for hand transcription
            index_manifest.json <- {i: page_id} so the mapping is never lost

    import: reads back any human_ocr_{i}.txt you've filled in and writes
            data/ocr_wer_gold/transcripts/{page_id}.txt, the exact layout
            eval/ocr_wer.py already scores against -- so nothing about the
            existing harness needs to change.

Usage:
    python -m data.pilot.index_ocr_gold export
    python -m data.pilot.index_ocr_gold import
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GOLD_DIR = Path("data/ocr_wer_gold")
PAGES_DIR = GOLD_DIR / "pages"
TRANSCRIPTS_DIR = GOLD_DIR / "transcripts"
PREDICTIONS_PATH = GOLD_DIR / "glm_ocr_predictions.json"
INDEXED_DIR = GOLD_DIR / "indexed"
INDEX_MANIFEST_PATH = INDEXED_DIR / "index_manifest.json"


def _load_index_manifest() -> dict[str, str]:
    if not INDEX_MANIFEST_PATH.exists():
        raise SystemExit(f"{INDEX_MANIFEST_PATH} missing -- run 'export' first.")
    return json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))


def export_indexed() -> None:
    page_paths = sorted(PAGES_DIR.glob("*.png"))
    if not page_paths:
        raise SystemExit(f"No pages found under {PAGES_DIR}/.")
    if not PREDICTIONS_PATH.exists():
        raise SystemExit(
            f"{PREDICTIONS_PATH} missing -- run data/pilot/glm_ocr_client.py first."
        )
    predictions = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))

    INDEXED_DIR.mkdir(parents=True, exist_ok=True)
    index_manifest: dict[str, str] = {}
    n_missing_pred = 0
    for i, page_path in enumerate(page_paths, 1):
        page_id = page_path.stem
        index_manifest[str(i)] = page_id

        pred_text = predictions.get(page_id, "")
        if not pred_text.strip():
            n_missing_pred += 1
        (INDEXED_DIR / f"ocr_model_{i}.txt").write_text(pred_text, encoding="utf-8")

        human_path = INDEXED_DIR / f"human_ocr_{i}.txt"
        if not human_path.exists():
            # Don't clobber transcription work already in progress.
            existing_transcript = TRANSCRIPTS_DIR / f"{page_id}.txt"
            seed = existing_transcript.read_text(encoding="utf-8") \
                if existing_transcript.exists() else ""
            human_path.write_text(seed, encoding="utf-8")

    INDEX_MANIFEST_PATH.write_text(
        json.dumps(index_manifest, indent=2), encoding="utf-8"
    )
    print(f"Exported {len(page_paths)} pages -> {INDEXED_DIR}/")
    print(f"  ocr_model_1..{len(page_paths)}.txt written "
          f"({n_missing_pred} empty/failed GLM-OCR predictions).")
    print(f"  human_ocr_1..{len(page_paths)}.txt ready for hand transcription "
          f"(open the matching page in {PAGES_DIR}/, e.g. "
          f"index_manifest.json['1'] -> its page image).")
    print(f"  Mapping written to {INDEX_MANIFEST_PATH}.")


def import_human_transcripts() -> None:
    index_manifest = _load_index_manifest()
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for i, page_id in index_manifest.items():
        human_path = INDEXED_DIR / f"human_ocr_{i}.txt"
        if not human_path.exists():
            continue
        text = human_path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        (TRANSCRIPTS_DIR / f"{page_id}.txt").write_text(text, encoding="utf-8")
        n_written += 1
    print(f"Wrote {n_written}/{len(index_manifest)} transcripts to {TRANSCRIPTS_DIR}/ "
          f"from filled-in human_ocr_*.txt files.")
    if n_written < len(index_manifest):
        print(f"  {len(index_manifest) - n_written} still blank -- "
          f"eval/ocr_wer.py needs all {len(index_manifest)} before it will run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "import"])
    args = parser.parse_args()
    if args.mode == "export":
        export_indexed()
    else:
        import_human_transcripts()


if __name__ == "__main__":
    main()
