"""Scanned-PDF detector + page-splitter for the OCR-WER gold set
(RESEARCH.md T5 / M1_ONBOARDING.md #11). This is prep tooling only -- it
does not run OCR and does not do the required hand transcription; a human
still has to write data/ocr_wer_gold/transcripts/{page_id}.txt for each page
this script produces.

Detection: a page is a scanned-image candidate if it has an embedded raster
image covering most of the page area. Checked this way, not via
smartlawai.core.ocr's DIGITAL_TEXT_MIN_CHARS text-length heuristic, because
some sources (e.g. the Indian High Court Judgments dataset) embed a
previously-run, often low-quality OCR text layer on top of the scanned
image -- pypdf's extract_text() then returns plenty of (garbled) text even
though the page is visually a scan, silently defeating a length-based
check. Confirmed by direct inspection this session: a genuine 1951 scan
(KLHC010000011951_1_1951-02-16.pdf) had a 612x1008 embedded JPEG exactly
matching its page's mediabox, while a genuinely digital "Proceeding Sheet"
page had zero embedded images -- image presence/coverage is the reliable
signal, text length is not, once a source has already been OCR'd once.

Usage:
    python -m eval.prepare_ocr_gold --input-dir data/pilot/ocr_raw_pdfs_hc
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pypdf import PdfReader

DEFAULT_NUM_PAGES = 50
DEFAULT_MAX_PER_DOC = 3
DEFAULT_DPI = 300
DEFAULT_SEED = 42
MIN_IMAGE_COVERAGE = 0.5  # embedded image must cover at least this fraction
                          # of the page area to count as a scan, not a logo


def find_scanned_candidates(input_dir: Path) -> list[tuple[Path, int]]:
    """Returns (pdf_path, 0-indexed page_number) for every page carrying an
    embedded image covering >= MIN_IMAGE_COVERAGE of the page area -- a
    scanned-page candidate, checked per-page since one PDF can mix digital
    and scanned pages."""
    candidates = []
    pdf_paths = sorted(input_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found under {input_dir}/ -- nothing to scan.")

    print(f"Scanning {len(pdf_paths)} PDFs for scanned-image pages...")
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as e:  # noqa: BLE001 - a corrupt/unreadable PDF, skip and report
            print(f"  skipping {pdf_path.name}: unreadable ({e})")
            continue
        for page_num, page in enumerate(reader.pages):
            mb = page.mediabox
            page_area = float(mb.width) * float(mb.height)
            if page_area <= 0:
                continue
            try:
                images = page.images
            except Exception:  # noqa: BLE001 - malformed image XObject, treat as no image
                images = []
            is_scan = any(
                img.image is not None
                and img.image.width * img.image.height >= MIN_IMAGE_COVERAGE * page_area
                for img in images
            )
            if is_scan:
                candidates.append((pdf_path, page_num))

    print(f"  found {len(candidates)} scanned-candidate pages across "
          f"{len(pdf_paths)} PDFs.")
    return candidates


def select_pages(candidates: list[tuple[Path, int]], num_pages: int,
                  max_per_doc: int, seed: int) -> list[tuple[Path, int]]:
    by_doc: dict[Path, list[int]] = {}
    for pdf_path, page_num in candidates:
        by_doc.setdefault(pdf_path, []).append(page_num)

    capped = [(pdf_path, page_num) for pdf_path, pages in by_doc.items()
              for page_num in pages[:max_per_doc]]

    if len(capped) < num_pages:
        raise SystemExit(
            f"Only {len(capped)} scanned-candidate pages available (after capping "
            f"{max_per_doc}/doc), need {num_pages}.\n"
            f"This is not something to fabricate or silently undershoot on -- widen "
            f"the source document set (e.g. add another year to "
            f"data/pilot/download_ocr_source_pdfs.py's YEARS) and re-run."
        )

    random.Random(seed).shuffle(capped)
    return capped[:num_pages]


def render_pages(selected: list[tuple[Path, int]], out_dir: Path,
                  dpi: int) -> dict[str, dict]:
    from pdf2image import convert_from_path

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for pdf_path, page_num in selected:
        page_id = f"{pdf_path.stem}_p{page_num + 1:03d}"
        images = convert_from_path(str(pdf_path), dpi=dpi,
                                    first_page=page_num + 1, last_page=page_num + 1)
        image_path = out_dir / f"{page_id}.png"
        images[0].save(image_path)
        manifest[page_id] = {"source_pdf": str(pdf_path), "page_number": page_num + 1}
        print(f"  {page_id} <- {pdf_path.name} p{page_num + 1}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                         help="Folder of source judgment PDFs to scan (searched recursively).")
    parser.add_argument("--num-pages", type=int, default=DEFAULT_NUM_PAGES)
    parser.add_argument("--max-per-doc", type=int, default=DEFAULT_MAX_PER_DOC,
                         help="Cap candidate pages taken from any single document, "
                              "so the gold set spans many documents, not one.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=Path("data/ocr_wer_gold/pages"))
    parser.add_argument("--manifest", type=Path,
                         default=Path("data/ocr_wer_gold/manifest.json"))
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"{args.input_dir} is not a directory.")

    candidates = find_scanned_candidates(args.input_dir)
    selected = select_pages(candidates, args.num_pages, args.max_per_doc, args.seed)

    print(f"Rendering {len(selected)} selected pages to {args.out_dir}/...")
    manifest = render_pages(selected, args.out_dir, args.dpi)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n=== DONE === {len(manifest)} pages -> {args.out_dir}/")
    print(f"Manifest written to {args.manifest}")
    print("\nNext steps:")
    print("  1. For each page, hand-transcribe it into "
          "data/ocr_wer_gold/transcripts/{page_id}.txt")
    print("  2. Run: python -m eval.ocr_wer")


if __name__ == "__main__":
    main()
