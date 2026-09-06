"""The real M1 Path 2 production run: Grok 4.1 Fast (Non-Reasoning) via the
Vertex AI OpenAI-compatible endpoint (global location), full summ corpus
(~7,130 docs, train+test). Winner of the E.3 model pilot (n=80, statistically
confirmed): 0.934 mean claim score, 1.8% hallucination, 96.9% section
acceptance -- see the plan doc's E.3.10 for the full comparison.

No batch tier exists for this model (confirmed against the live Vertex
pricing page) -- this runs on-demand, parallelized. Verified real quota
(Console Quotas page, not the model card's earlier-reported numbers which
were wrong): 40 QPM / 220,000 input-TPM / 10,000 output-TPM per project.
Output-TPM is binding at our ~929 avg output tokens/doc -> ~10.8 docs/min
sustainable, needing only ~1.7 concurrent workers (Little's Law) -- an
earlier attempt at 100 workers oversubscribed this ~60x and was stopped
after 126 docs once the real quota was found. 3 workers gives headroom
without the earlier waste. Expected total runtime: ~11 hours. Periodic
auth-token refresh included regardless, since a run this long would hit the
~1hr token expiry multiple times over.

Output goes straight to JSONL matching data/score_filter_pairs.py's expected
schema (doc_id, source_headnote, generated_text per section) so the HHEM
filter can run directly on this without a reshaping step.
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import google.auth
import google.auth.transport.requests
import requests
from datasets import load_from_disk

import ctypes
import sys

# Prevent Windows from sleeping (system + display) for the duration of this
# process -- an ~11hr unattended overnight run would otherwise stall (or worse,
# drop network mid-request) the moment the laptop goes idle. Released
# automatically on process exit (normal or crash) since it's tied to this
# process's thread state, not a persistent system setting.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
if sys.platform == "win32":
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )

MODEL = "xai/grok-4.1-fast-non-reasoning"
URL = "https://aiplatform.googleapis.com/v1/projects/smartlawai-1/locations/global/endpoints/openapi/chat/completions"
OUT_DIR = Path("data/processed/path2_raw")
MAX_WORKERS = 3  # real verified quota: 40 QPM / 220k input-TPM / 10k output-TPM
                 # -- output-TPM is binding at ~10.8 docs/min, needing only ~1.7
                 # concurrent workers (Little's Law); 3 gives headroom without
                 # oversubscribing the way 100 did (~90%+ throttled, wasted retries)
TOKEN_REFRESH_INTERVAL_S = 45 * 60  # refresh well before the ~1hr token expiry

PROMPT_TEMPLATE = """You are given the full text of an Indian Supreme Court judgment and its
existing professional headnote summary. Your task is NOT to write a new summary --
it is to reorganize the EXISTING headnote content into these five sections:
facts, statute, argument, analysis, judgement.

Rules:
- Use ONLY sentences/content already present in the headnote below. Do not invent
  new facts, holdings, or citations not already stated in the headnote.
- You may lightly rephrase for clarity within a section, but the substance of every
  claim must trace back to the headnote text.
- If the headnote has no content for a given section, return an empty string for it.
- Return strict JSON only, matching this schema:
  {{"facts": "...", "statute": "...", "argument": "...", "analysis": "...", "judgement": "..."}}

JUDGMENT TEXT:
{document}

EXISTING HEADNOTE:
{summary}
"""


class TokenManager:
    """Thread-safe access token that refreshes itself periodically so a
    multi-hour run doesn't start failing with 401s after ~1hr."""

    def __init__(self):
        self._lock = threading.Lock()
        self._creds, _ = google.auth.default()
        self._last_refresh = 0
        self._refresh()

    def _refresh(self):
        self._creds.refresh(google.auth.transport.requests.Request())
        self._last_refresh = time.time()

    def get(self) -> str:
        with self._lock:
            if time.time() - self._last_refresh > TOKEN_REFRESH_INTERVAL_S:
                self._refresh()
            return self._creds.token


def generate_one(token_mgr: TokenManager, doc: dict, split: str) -> dict:
    document_text = " ".join(doc["document"])
    summary_text = " ".join(doc["summary"])
    prompt = PROMPT_TEMPLATE.format(document=document_text, summary=summary_text)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    last_err = None
    for attempt in range(6):
        headers = {"Authorization": f"Bearer {token_mgr.get()}", "Content-Type": "application/json"}
        try:
            resp = requests.post(URL, headers=headers, json=body, timeout=180)
            resp.raise_for_status()
            raw = resp.json()
            content = raw["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                    parsed = parsed[0]
                parse_ok = isinstance(parsed, dict)
            except Exception:
                parsed, parse_ok = {}, False
            if not parse_ok:
                return {"doc_id": doc["id"], "split": split, "error": "unparseable_json",
                        "raw_content": content[:2000]}

            pairs = []
            for section, text in parsed.items():
                if not text:
                    continue
                pairs.append({
                    "doc_id": doc["id"],
                    "split": split,
                    "section": section,
                    "source_headnote": summary_text,
                    "generated_text": text,
                    "provenance": "path2-silver-grok-4.1-fast-non-reasoning",
                })
            return {"doc_id": doc["id"], "split": split, "pairs": pairs}
        except Exception as e:
            last_err = e
            if "429" in str(e) and attempt < 5:
                time.sleep(min(60, 5 * (2 ** attempt)))
            elif attempt < 5:
                time.sleep(5)
            else:
                break
    return {"doc_id": doc["id"], "split": split, "error": str(last_err)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summ = load_from_disk("data/raw/summ")

    # Only count genuine successes as "done" -- an error record (from a crash,
    # network drop, or persistent 429 mid-run) must NOT block a retry on the
    # next restart. Overnight unattended runs need failures to self-heal via
    # the outer restart loop, not get silently stuck as permanently skipped.
    already_done = set()
    for p in OUT_DIR.glob("*.json"):
        if not p.stem.isdigit():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                if "error" not in json.load(f):
                    already_done.add(int(p.stem))
        except Exception:
            pass  # unreadable/partial file -- treat as not done, retry it
    todo = []
    for split in ["train", "test"]:
        for row in summ[split]:
            if row["id"] not in already_done:
                todo.append((row, split))

    print(f"Total docs: {sum(len(summ[s]) for s in ['train','test'])}, "
          f"already done: {len(already_done)}, generating: {len(todo)}")

    token_mgr = TokenManager()
    t_start = time.time()
    n_pairs_total = 0
    n_errors = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(generate_one, token_mgr, doc, split): (doc, split)
                   for doc, split in todo}
        for fut in as_completed(futures):
            doc, split = futures[fut]
            record = fut.result()
            completed += 1
            with open(OUT_DIR / f"{doc['id']}.json", "w", encoding="utf-8") as f:
                json.dump(record, f)
            if "error" in record:
                n_errors += 1
                print(f"  [{completed}/{len(todo)}] doc_id={doc['id']} ERROR: "
                      f"{record['error'][:100]}")
            else:
                n_pairs_total += len(record["pairs"])
            if completed % 100 == 0 or completed == len(todo):
                elapsed_min = (time.time() - t_start) / 60
                rate = completed / elapsed_min if elapsed_min > 0 else 0
                print(f"  progress: {completed}/{len(todo)} docs, {n_errors} errors, "
                      f"{n_pairs_total} pairs so far, {elapsed_min:.1f}min elapsed, "
                      f"{rate:.1f} docs/min")

    elapsed_min = (time.time() - t_start) / 60
    print(f"\n=== DONE: {completed} docs processed, {n_errors} errors, "
          f"{n_pairs_total} section pairs, {elapsed_min:.1f} minutes total ===")

    # Consolidate into the single raw JSONL build_pairs.py-shaped output
    out_path = Path("data/processed/section_pairs_raw.jsonl")
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for p in sorted(OUT_DIR.glob("*.json"), key=lambda x: int(x.stem)):
            with open(p, encoding="utf-8") as f:
                record = json.load(f)
            for pair in record.get("pairs", []):
                out_f.write(json.dumps(pair) + "\n")
                n_written += 1
    print(f"Wrote {n_written} section pairs to {out_path}")


if __name__ == "__main__":
    main()
