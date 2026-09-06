"""Thin client for GLM-OCR served locally via Ollama, plus a batch-predict
mode that runs it over the OCR-WER gold set's 50 scanned pages and caches
the results -- so predictions exist before the human transcriptions do
(the "scan first" half of the plan; scoring against the transcripts once
they exist happens in eval/ocr_wer.py's --engine glm-ocr mode).

Requires the user to have run `ollama pull glm-ocr` themselves first
(Ollama must be installed and its local service running on
http://localhost:11434, the default).

Uses Ollama's native /api/generate endpoint, not the OpenAI-compatible
chat endpoint -- the latter has known limitations for vision requests
with Ollama (confirmed via research, not assumed).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "glm-ocr"
PROMPT = "Transcribe all text in this image exactly as it appears, preserving line breaks. Output only the transcribed text, nothing else."
TIMEOUT_S = 120

GOLD_PAGES_DIR = Path("data/ocr_wer_gold/pages")
PREDICTIONS_PATH = Path("data/ocr_wer_gold/glm_ocr_predictions.json")


def ocr_image(image_path: Path) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "images": [image_b64],
            "stream": False,
        },
        timeout=TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()["response"]


def run_batch(pages_dir: Path = GOLD_PAGES_DIR,
              out_path: Path = PREDICTIONS_PATH) -> dict[str, str]:
    page_paths = sorted(pages_dir.glob("*.png"))
    if not page_paths:
        raise SystemExit(f"No PNG pages found under {pages_dir}/ -- nothing to run GLM-OCR on.")

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
