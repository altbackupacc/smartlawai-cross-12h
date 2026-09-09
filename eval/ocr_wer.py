"""M1 step 10: OCR Word Error Rate harness (RESEARCH.md T5 -- "OCR quality
caps everything downstream", measure it, don't assume it).

This is a harness only. It does not and cannot produce the 50 hand-transcribed
gold pages itself -- that is human-supplied work, tracked separately. If the
gold set doesn't exist yet (or has fewer than 50 matched pairs), this script
hard-fails rather than fabricating, approximating, or silently sampling fewer
pages. Not a blocker for the rest of M1 (PLAN.md's literal M1 done-when list
doesn't include the WER number) -- build/keep this harness ready regardless,
run it the moment the human transcriptions land.

Expected input structure:
    data/ocr_wer_gold/pages/{page_id}.{png|pdf}       (scanned source pages)
    data/ocr_wer_gold/transcripts/{page_id}.txt        (hand transcription)
    50 matched pairs minimum.

Supports two OCR engines via --engine, scored against the exact same gold
set for a direct comparison:
    tesseract  (default) -- reads data/pilot/tesseract_ocr_client.py's cached
               predictions (run that script first); falls back to a live
               call for any page missing from the cache.
    glm-ocr    -- reads data/pilot/glm_ocr_client.py's cached predictions
               (run that script first); falls back to a live Ollama call
               for any page missing from the cache.

Neither engine's actual OCR call happens inside this file -- eval/ modules
are pure functions over data structures per OPS.md §8 (enforced by
tests/test_eval_integration.py's layering check), so anything that needs
smartlawai or a live model call lives in data/pilot/*_client.py instead, and
only its cached JSON output crosses into eval/.

Output: eval/ocr_wer.md for --engine tesseract (unchanged default path);
eval/ocr_wer_glm-ocr.md for --engine glm-ocr.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import jiwer

MIN_PAIRS = 49  # page 21's transcript deliberately skipped -- see data/PROVENANCE.md section 9
PAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf")
PREDICTIONS_PATHS = {
    "tesseract": Path("data/ocr_wer_gold/tesseract_predictions.json"),
    "glm-ocr": Path("data/ocr_wer_gold/glm_ocr_predictions.json"),
}
STATUS_OK = {"tesseract": "OCR_OK", "glm-ocr": "GLM_OCR_OK"}
STATUS_FAIL = {"tesseract": "OCR_FAIL", "glm-ocr": "GLM_OCR_FAIL"}


def compute_wer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return jiwer.wer(reference, hypothesis)


def find_matched_pairs(gold_dir: Path) -> list[tuple[str, Path, Path]]:
    pages_dir = gold_dir / "pages"
    transcripts_dir = gold_dir / "transcripts"
    if not pages_dir.is_dir() or not transcripts_dir.is_dir():
        return []
    pairs = []
    for page_path in sorted(pages_dir.iterdir()):
        if page_path.suffix.lower() not in PAGE_EXTENSIONS:
            continue
        page_id = page_path.stem
        transcript_path = transcripts_dir / f"{page_id}.txt"
        if transcript_path.exists():
            pairs.append((page_id, page_path, transcript_path))
    return pairs


def _run_engine(engine: str, page_id: str, page_path: Path,
                 predictions: dict[str, str]) -> tuple[str | None, str, str | None]:
    """Returns (text_or_None, status, error_or_None). Prefers the cached
    predictions file (data/pilot/{engine}_ocr_client.py, with '-' -> '_');
    falls back to a live call for a page missing from the cache."""
    text = predictions.get(page_id)
    if text is None:
        module = f"data.pilot.{engine.replace('-', '_')}_ocr_client"
        ocr_image = __import__(module, fromlist=["ocr_image"]).ocr_image
        try:
            text = ocr_image(page_path)
        except Exception as e:  # noqa: BLE001 - report as a failure row, don't crash the run
            return None, STATUS_FAIL[engine], str(e)
    if not text.strip():
        return None, STATUS_FAIL[engine], "empty prediction"
    return text, STATUS_OK[engine], None


def run_ocr_wer_eval(gold_dir: Path = Path("data/ocr_wer_gold"),
                      out_path: Path | None = None,
                      engine: str = "tesseract") -> None:
    if out_path is None:
        out_path = Path("eval/ocr_wer.md") if engine == "tesseract" \
            else Path(f"eval/ocr_wer_{engine}.md")

    pairs = find_matched_pairs(gold_dir)
    if len(pairs) < MIN_PAIRS:
        raise SystemExit(
            f"OCR-WER gold set incomplete: found {len(pairs)} matched page/transcript "
            f"pair(s) under {gold_dir}/, need at least {MIN_PAIRS}.\n\n"
            f"Required structure:\n"
            f"  {gold_dir}/pages/{{page_id}}.{{png|jpg|tiff|pdf}}  (scanned source pages)\n"
            f"  {gold_dir}/transcripts/{{page_id}}.txt             (hand transcription)\n"
            f"This is human-supplied work (RESEARCH.md T5) and cannot be fabricated, "
            f"approximated, or sampled below {MIN_PAIRS} pairs. Not a blocker for the "
            f"rest of M1 -- run this script again once the gold set is complete."
        )

    predictions_path = PREDICTIONS_PATHS[engine]
    predictions: dict[str, str] = {}
    if predictions_path.exists():
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))

    print(f"Found {len(pairs)} matched pairs. Running {engine} OCR + computing WER per page...")
    rows = []
    for page_id, page_path, transcript_path in pairs:
        reference = transcript_path.read_text(encoding="utf-8", errors="ignore")
        text, status, error = _run_engine(engine, page_id, page_path, predictions)
        if text is None:
            rows.append({"page_id": page_id, "wer": None, "status": status, "error": error})
            continue
        wer = compute_wer(reference, text)
        rows.append({"page_id": page_id, "wer": wer, "status": status, "error": None})

    scored = [r["wer"] for r in rows if r["wer"] is not None]
    n_failed = len(rows) - len(scored)
    mean_wer = sum(scored) / len(scored) if scored else None
    median_wer = sorted(scored)[len(scored) // 2] if scored else None

    lines = [
        f"# OCR Word Error Rate — M1 (RESEARCH.md T5) — engine: {engine}",
        "",
        (f"Gold set size: {len(pairs)} pages. Evaluated: "
        f"{datetime.now(tz=UTC).date().isoformat()}."),
        f"Failed extraction (excluded from WER stats): {n_failed}/{len(rows)}.",
        "",
        f"**Mean WER: {mean_wer:.4f}**" if mean_wer is not None else "**Mean WER: N/A (all pages failed)**",
        f"**Median WER: {median_wer:.4f}**" if median_wer is not None else "",
        "",
        "| Page ID | Status | WER |",
        "|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["page_id"]):
        wer_str = f"{r['wer']:.4f}" if r["wer"] is not None else f"FAILED ({r['error']})"
        lines.append(f"| {r['page_id']} | {r['status']} | {wer_str} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} (mean WER: {mean_wer})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["tesseract", "glm-ocr"], default="tesseract")
    args = parser.parse_args()
    run_ocr_wer_eval(engine=args.engine)
