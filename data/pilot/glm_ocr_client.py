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


ANCHOR_LEN = 40
# Only used for the FLAG (not auto-edit) check below -- see its docstring for
# why a large threshold is required and why this is detection-only.
FLAG_REPEAT_LEN = 150


def _dedupe_trailing_loop(text: str, anchor_len: int = ANCHOR_LEN) -> str:
    """glm-ocr has no calibrated stop token for a page's true end-of-content
    (confirmed against a physically torn-off gold page: the model transcribed
    every real line correctly, then -- instead of stopping -- restarted from
    its own opening line verbatim, with no blank-line separator, so a
    paragraph-boundary split doesn't catch it). Search for the response's own
    first `anchor_len` characters reappearing later and cut there -- a direct
    restart-from-the-top loop, not new content."""
    if len(text) <= anchor_len:
        return text
    anchor = text[:anchor_len]
    repeat_at = text.find(anchor, anchor_len)
    if repeat_at == -1:
        return text
    return text[:repeat_at].rstrip()


def detect_possible_hallucinated_tail(text: str,
                                       min_len: int = FLAG_REPEAT_LEN) -> str | None:
    """FLAGS (never edits) a page whose output contains a long exact repeated
    block not anchored at position 0. One real gold page turned out to have
    exactly this shape (a correct transcription followed by a paraphrased
    "recap" that was itself duplicated) -- but a first attempt at
    auto-truncating on this signal also fired on a second real gold page
    where the *source document itself* genuinely repeats a long clause
    (interest recalculation language spanning two paragraphs) worded closely
    enough that the sliding match caught it too, and cut into real content
    that has nothing to do with model looping. Auto-editing on this signal is
    provably unsafe (verified against both cases) -- so this only returns the
    matched excerpt for a human to judge on the actual page image, never
    modifies the transcript."""
    n = len(text)
    for start in range(min_len, n - min_len):
        window = text[start:start + min_len]
        first_at = text.find(window)
        if first_at != -1 and first_at < start:
            return text[first_at:start + min_len]
    return None


def _strip_leaked_prompt(text: str) -> str:
    """On a near-blank page (confirmed against a real gold page that's just a
    cover leaf with two lines of text), glm-ocr runs out of real content and
    fills the rest of its budget by echoing the instruction prompt itself
    back as if it were transcribed text. Cut it off there -- it is never
    genuine page content."""
    leak_at = text.find(PROMPT)
    if leak_at == -1:
        return text
    return text[:leak_at].rstrip()


def ocr_image(image_path: Path) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    # Confirmed via Ollama's server.log: the prompt+image alone is ~4053 tokens,
    # so the default (small) context window truncated generation almost
    # immediately (done_reason: "length" after ~40 tokens). num_ctx=8192 gives
    # real headroom to generate the page. Non-streaming mode returned "done":
    # False for this model/version (a server-side quirk, not a client bug) --
    # streaming sidesteps it since we just read until the connection closes,
    # rather than depending on a final done=true event.
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "images": [image_b64],
            "stream": True,
            "options": {"num_predict": 2048, "num_ctx": 8192},
        },
        timeout=TIMEOUT_S,
        stream=True,
    )
    response.raise_for_status()
    chunks = []
    for line in response.iter_lines():
        if not line:
            continue
        chunks.append(json.loads(line).get("response", ""))
    text = _dedupe_trailing_loop("".join(chunks))
    return _strip_leaked_prompt(text)


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
