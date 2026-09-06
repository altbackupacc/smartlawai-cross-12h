"""Downloads real old-judgment PDFs for the OCR-WER gold set from the
Indian High Court Judgments public dataset (AWS Open Data, no credentials
needed -- see https://github.com/vanga/indian-high-court-judgments).

Supersedes download_ocr_source_pdfs.py's Kaggle source: that dataset turned
out to be Indian Kanoon's own clean re-transcribed text (never scanned),
so every page had extractable text regardless of the document's age --
0/61 scanned candidates found in the 1950 sample. This High Court dataset
is scraped directly from the eCourts portal (closer to the original filing),
and its own docs confirm some files are genuinely scanned.

Important gotcha, found by inspecting the bucket directly (not documented
anywhere): the S3 `year=<N>` partition is the case's FILING year (parsed
from the case number), not the decision/order date -- e.g.
`data/tar/year=1950/.../HCBM020000041950_1_2006-11-21.pdf` is an order
from 2006 for a case numbered in 1950. Filtering by that partition alone
would silently pull modern digital documents. Filenames are
`{case_code}{year}_{part}_{order_date}.pdf`; this script downloads a
handful of small year= partitions (whole range is ~20MB, checked directly
against the bucket before writing this) then filters by the trailing
ORDER DATE, not the partition, to find genuinely old documents.
"""
from __future__ import annotations

import io
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

BUCKET_URL = "https://indian-high-court-judgments.s3.amazonaws.com"
YEAR_PARTITIONS = range(1947, 1966)  # case-filing-year partitions to scan (~20MB total)
MAX_ORDER_YEAR = 1970  # keep only files whose actual ORDER DATE is this year or earlier
OUT_DIR = Path("data/pilot/ocr_raw_pdfs_hc")
FILENAME_DATE_RE = re.compile(r"_(\d{4})-\d{2}-\d{2}\.pdf$")
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_tar_keys() -> list[str]:
    keys = []
    for year in YEAR_PARTITIONS:
        r = requests.get(f"{BUCKET_URL}/", params={
            "list-type": "2", "prefix": f"data/tar/year={year}/", "max-keys": "1000"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall("s3:Contents", NS):
            key = c.find("s3:Key", NS).text
            if key.endswith("data.tar"):
                keys.append(key)
    return keys


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Listing data.tar files across year={YEAR_PARTITIONS.start}-"
          f"{YEAR_PARTITIONS.stop - 1} partitions...")
    keys = list_tar_keys()
    print(f"  found {len(keys)} tar archives.")

    kept, skipped = 0, 0
    for key in keys:
        print(f"Downloading {key}...")
        resp = requests.get(f"{BUCKET_URL}/{key}")
        resp.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(resp.content)) as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".pdf"):
                    continue
                m = FILENAME_DATE_RE.search(member.name)
                if not m or int(m.group(1)) > MAX_ORDER_YEAR:
                    skipped += 1
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                out_path = OUT_DIR / Path(member.name).name
                out_path.write_bytes(extracted.read())
                kept += 1

    print(f"\nDone: {kept} PDFs with order date <= {MAX_ORDER_YEAR} kept -> {OUT_DIR}")
    print(f"  ({skipped} files skipped: order date after {MAX_ORDER_YEAR})")
    print("Next: python -m eval.prepare_ocr_gold --input-dir data/pilot/ocr_raw_pdfs_hc")


if __name__ == "__main__":
    main()
