#!/bin/bash
# Auto-retry wrapper for download_ocr_source_pdfs.py -- the Kaggle/GCS-backed
# download has proven flaky (ConnectionError: Read timed out after 26MB on
# the first attempt), but kagglehub resumes from the partial .archive file
# rather than restarting, so retrying is cheap. Same pattern as
# data/run_path2_supervised.sh used for the Path 2 production run.
export PATH="$PATH:/c/Users/kasir/AppData/Local/Microsoft/WinGet/Packages/oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe/poppler-25.07.0/Library/bin"
cd "C:\Users\kasir\OneDrive\Documents\smartlawai-1" || exit 1

MAX_RESTARTS=30
attempt=0
until ".venv/Scripts/python.exe" -u data/pilot/download_ocr_source_pdfs.py; do
  attempt=$((attempt + 1))
  echo "=== attempt $attempt failed, retrying in 15s (max $MAX_RESTARTS) ==="
  if [ "$attempt" -ge "$MAX_RESTARTS" ]; then
    echo "=== giving up after $MAX_RESTARTS attempts ==="
    exit 1
  fi
  sleep 15
done
echo "=== download completed successfully ==="
