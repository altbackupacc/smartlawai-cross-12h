"""Downloads real scanned-candidate judgment PDFs for the OCR-WER gold set
(M1_ONBOARDING.md #11 / RESEARCH.md T5).

Source: "Legal Dataset: SC Judgments India (1950-2024)" (Kaggle, CC0),
organized as one directory per year -- 76 years, 26,000 PDFs total.

Per-year targeted download (kagglehub's `path=` argument) turned out not to
work against this specific dataset: Kaggle stores it as a single ~6.4GB
opaque archive blob ("1.archive"), not as individually-listable files, so
`dataset_download(path="1950")` 404s regardless of the exact path format --
that's a property of how this dataset was uploaded, not a kagglehub bug or
something a different path string fixes. The only way to get at any of it
is the full download; this script does that once, then filters to the
requested YEARS locally from the extracted cache.

1950 is the starting year: the Supreme Court sat with only 8 judges that
year (its first full year of operation) and every judgment from that decade
was necessarily typewritten (word processors didn't exist yet), so any
digital PDF of one today is unavoidably a scan of an old paper document --
exactly what the OCR-WER stress test needs. If 1950 alone doesn't yield
enough scanned-candidate pages once eval/prepare_ocr_gold.py runs, widening
YEARS costs nothing extra -- the full archive is already cached locally by
kagglehub after the first run.

Requires the user's own Kaggle API credentials (same as the classic CLI):
~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY env vars.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATASET = "adarshsingh0903/legal-dataset-sc-judgments-india-19502024"
YEARS = [1950]
OUT_DIR = Path("data/pilot/ocr_raw_pdfs")
# The dataset extracts with one wrapping subfolder before the year folders
# (confirmed by inspecting the actual cached download -- not documented on
# the Kaggle page itself, which only describes the year-folder layer).
YEAR_FOLDERS_SUBDIR = "supreme_court_judgments"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading the full dataset (~6.4GB, one-time) from {DATASET}...")
    cached_root = Path(kagglehub.dataset_download(DATASET)) / YEAR_FOLDERS_SUBDIR
    print(f"  cached at {cached_root}")

    for year in YEARS:
        source_year_dir = cached_root / str(year)
        if not source_year_dir.is_dir():
            print(f"  WARNING: {source_year_dir} not found in cached dataset -- "
                  f"skipping {year}")
            continue
        dest = OUT_DIR / str(year)
        shutil.copytree(source_year_dir, dest, dirs_exist_ok=True)
        n = len(list(dest.glob("*.pdf")))
        print(f"  {year}: {n} PDFs -> {dest}")

    total = sum(len(list((OUT_DIR / str(y)).glob("*.pdf"))) for y in YEARS
                if (OUT_DIR / str(y)).is_dir())
    print(f"\nDone: {total} PDFs across {len(YEARS)} year(s) -> {OUT_DIR}")
    print("Next: python -m eval.prepare_ocr_gold --input-dir data/pilot/ocr_raw_pdfs")


if __name__ == "__main__":
    main()
