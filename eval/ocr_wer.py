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

Output: eval/ocr_wer.md (markdown table + summary stats + gold-set size + date)
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import jiwer

from smartlawai.core.ocr import extract_text

MIN_PAIRS = 50
PAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf")


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


def run_ocr_wer_eval(gold_dir: Path = Path("data/ocr_wer_gold"),
                      out_path: Path = Path("eval/ocr_wer.md")) -> None:
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

    print(f"Found {len(pairs)} matched pairs. Running OCR + computing WER per page...")
    rows = []
    for page_id, page_path, transcript_path in pairs:
        reference = transcript_path.read_text(encoding="utf-8", errors="ignore")
        result = extract_text(str(page_path))
        if result.ocr_status == "OCR_FAIL":
            rows.append({"page_id": page_id, "wer": None, "status": "OCR_FAIL",
                         "error": result.error})
            continue
        wer = compute_wer(reference, result.text)
        rows.append({"page_id": page_id, "wer": wer, "status": result.ocr_status,
                     "error": None})

    scored = [r["wer"] for r in rows if r["wer"] is not None]
    n_failed = len(rows) - len(scored)
    mean_wer = sum(scored) / len(scored) if scored else None
    median_wer = sorted(scored)[len(scored) // 2] if scored else None

    lines = [
        "# OCR Word Error Rate — M1 (RESEARCH.md T5)",
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
    run_ocr_wer_eval()
