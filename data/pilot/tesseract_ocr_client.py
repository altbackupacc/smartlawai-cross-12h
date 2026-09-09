"""Thin wrapper around the serving pipeline's Tesseract engine
(smartlawai.core.ocr.extract_text), plus a batch-predict mode over the
OCR-WER gold set's 50 scanned pages, caching results the same way
glm_ocr_client.py does.

This script exists so eval/ocr_wer.py never has to import `smartlawai`
directly -- eval/ modules are pure functions over data structures per
OPS.md §8 (enforced by tests/test_eval_integration.py's layering check).
Everything that needs the actual pipeline/adapters code lives here, outside
eval/, and only its cached JSON output crosses that boundary.

Usage:
    python -m data.pilot.tesseract_ocr_client
"""
from __future__ import annotations

import json
from pathlib import Path

GOLD_PAGES_DIR = Path("data/ocr_wer_gold/pages")
PREDICTIONS_PATH = Path("data/ocr_wer_gold/tesseract_predictions.json")


def ocr_image(image_path: Path) -> str:
    from smartlawai.core.ocr import extract_text
    result = extract_text(str(image_path))
    if result.ocr_status == "OCR_FAIL":
        raise RuntimeError(result.error or "tesseract OCR failed")
    return result.text


def run_batch(pages_dir: Path = GOLD_PAGES_DIR,
              out_path: Path = PREDICTIONS_PATH) -> dict[str, str]:
    page_paths = sorted(pages_dir.glob("*.png"))
    if not page_paths:
        raise SystemExit(f"No PNG pages found under {pages_dir}/ -- nothing to run Tesseract on.")

    predictions: dict[str, str] = {}
    if out_path.exists():
        predictions = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"Resuming: {len(predictions)} predictions already cached in {out_path}.")

    for i, page_path in enumerate(page_paths, 1):
        page_id = page_path.stem
        if page_id in predictions:
            continue
        print(f"  [{i}/{len(page_paths)}] {page_id}...")
        try:
            predictions[page_id] = ocr_image(page_path)
        except Exception as e:  # noqa: BLE001 - log and continue, don't lose earlier progress
            print(f"    FAILED: {e}")
            predictions[page_id] = ""
        out_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    n_empty = sum(1 for v in predictions.values() if not v.strip())
    print(f"\nDone: {len(predictions)} predictions -> {out_path} ({n_empty} empty/failed)")
    return predictions


if __name__ == "__main__":
    run_batch()
