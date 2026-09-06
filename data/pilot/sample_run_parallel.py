"""Parallelized sample-document run against Gemini 3.8 Flash / Mistral Medium 3,
for a bigger (n=80) faithfulness comparison than the original n=10 pilot.

Generation calls go to Google's/Mistral's hosted infra, not our L4 -- the L4
is only used for HHEM scoring afterward. Parallelizing here only speeds up
the many-sequential-HTTP-calls bottleneck in the original pilot scripts.
Modest concurrency (8 workers) + retry-with-backoff on 429s, since Gemini's
actual concurrent-request rate limit hasn't been tested before now.

Reuses the original 10 docs' already-saved results (same seed=42 sample) and
generates 70 new ones (seed=43, excluding the original 10) to reach n=80,
rather than re-spending on docs already scored.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_from_disk
from google import genai
from google.genai import types

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

ORIGINAL_N = 10
TARGET_N = 80
ORIGINAL_SEED = 42
EXTRA_SEED = 43
MAX_WORKERS = 8


def generate_one(client, model: str, location: str, doc: dict) -> dict:
    document_text = " ".join(doc["document"])
    summary_text = " ".join(doc["summary"])
    prompt = PROMPT_TEMPLATE.format(document=document_text, summary=summary_text)

    t0 = time.time()
    last_err = None
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.0
                ),
            )
            elapsed = time.time() - t0
            try:
                parsed = json.loads(resp.text)
                parse_ok = isinstance(parsed, dict)
                if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                    parsed, parse_ok = parsed[0], True  # same list-wrap quirk as 3.1 Pro
            except Exception:
                parsed, parse_ok = {"_raw": resp.text}, False
            usage = resp.usage_metadata
            return {
                "doc_id": doc["id"],
                "il_tur_doc_tokens": doc["num_doc_tokens"],
                "il_tur_summ_tokens": doc["num_summ_tokens"],
                "prompt_token_count": usage.prompt_token_count,
                "candidates_token_count": usage.candidates_token_count,
                "elapsed_s": round(elapsed, 1),
                "parse_ok": parse_ok,
                "sections": parsed,
            }
        except Exception as e:
            last_err = e
            if "429" in str(e) and attempt < 3:
                time.sleep(10 * (attempt + 1))
            else:
                break
    return {"doc_id": doc["id"], "error": str(last_err)}


def main(model: str, location: str, out_dir_name: str) -> None:
    summ = load_from_disk("data/raw/summ")
    train = summ["train"]

    original_idxs = random.Random(ORIGINAL_SEED).sample(range(len(train)), ORIGINAL_N)
    original_ids = {train[i]["id"] for i in original_idxs}

    pool = [i for i in range(len(train)) if train[i]["id"] not in original_ids]
    extra_idxs = random.Random(EXTRA_SEED).sample(pool, TARGET_N - ORIGINAL_N)

    out_dir = Path("data/pilot") / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse already-saved results for the original 10 if present in the existing
    # (non-parallel) pilot output dir for this model.
    reuse_dir = Path("data/pilot") / model.replace("/", "_")
    results = []
    to_generate = []
    for i in original_idxs:
        doc = train[i]
        existing = reuse_dir / f"{doc['id']}.json"
        if existing.exists():
            with open(existing, encoding="utf-8") as f:
                results.append(json.load(f))
        else:
            to_generate.append(doc)
    to_generate.extend(train[i] for i in extra_idxs)

    print(f"Reusing {len(results)} existing results, generating {len(to_generate)} new "
          f"({MAX_WORKERS} parallel workers)...")

    client = genai.Client(vertexai=True, project="smartlawai-1", location=location)

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool_exec:
        futures = {pool_exec.submit(generate_one, client, model, location, doc): doc
                   for doc in to_generate}
        for fut in as_completed(futures):
            doc = futures[fut]
            record = fut.result()
            results.append(record)
            completed += 1
            status = "OK" if "error" not in record else f"ERROR: {record['error'][:80]}"
            print(f"  [{completed}/{len(to_generate)}] doc_id={doc['id']} -> {status}")
            with open(out_dir / f"{doc['id']}.json", "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

    ok = [r for r in results if "error" not in r]
    n_parsed = sum(1 for r in ok if r["parse_ok"])
    if ok:
        in_ratio = sum(r["prompt_token_count"] for r in ok) / sum(
            r["il_tur_doc_tokens"] + r["il_tur_summ_tokens"] for r in ok)
        out_ratio = sum(r["candidates_token_count"] for r in ok) / sum(
            r["il_tur_summ_tokens"] for r in ok)
        print("\n=== AGGREGATE ===")
        print(f"Samples: {len(ok)}/{len(results)} succeeded, {n_parsed}/{len(ok)} valid JSON dict")
        print(f"Input ratio: {in_ratio:.3f}  Output ratio: {out_ratio:.3f}")

    with open(out_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} total results to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--location", default="global")
    parser.add_argument("--out-dir", default="gemini-3.8-flash-n80")
    args = parser.parse_args()
    main(args.model, args.location, args.out_dir)
