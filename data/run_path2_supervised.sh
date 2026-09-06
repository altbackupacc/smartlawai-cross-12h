#!/bin/bash
# Auto-restart supervisor for the overnight Path 2 production run.
# If the python process crashes (network drop, unhandled exception, etc.),
# relaunch it -- the resume logic in run_path2_production.py skips genuine
# successes and retries anything else, so a crash-restart just continues
# rather than losing progress or requiring a 3am manual fix.
cd "C:\Users\kasir\OneDrive\Documents\smartlawai-1"
MAX_RESTARTS=15
count=0
until ./.venv/Scripts/python.exe data/run_path2_production.py; do
  count=$((count+1))
  echo "=== run_path2_production.py exited non-zero (restart $count/$MAX_RESTARTS) ==="
  if [ "$count" -ge "$MAX_RESTARTS" ]; then
    echo "=== hit max restarts, giving up -- needs manual attention ==="
    break
  fi
  sleep 30
done
echo "=== supervisor loop ended ==="
