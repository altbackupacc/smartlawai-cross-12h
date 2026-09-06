"""n=80 run against Grok 4.1 Fast (Non-Reasoning) via the OpenAI-compatible
Vertex endpoint (global location). Last model in the comparison. Parallelized
(non-reasoning should be fast with no hidden reasoning-token overhead), with
retry-with-backoff on 429s since this model's actual concurrent quota is
untested.
"""
from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import google.auth
import google.auth.transport.requests
import requests
from datasets import load_from_disk

MODEL = "xai/grok-4.1-fast-non-reasoning"
URL = "https://aiplatform.googleapis.com/v1/projects/smartlawai-1/locations/global/endpoints/openapi/chat/completions"
OUT_DIR_NAME = "grok-4.1-fast-non-reasoning-n80"
N_SAMPLES = 80
SEED = 42
MAX_WORKERS = 3

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


def get_token() -> str:
    creds, _ = google.auth.default()
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def generate_one(headers: dict, doc: dict) -> dict:
    document_text = " ".join(doc["document"])
    summary_text = " ".join(doc["summary"])
    prompt = PROMPT_TEMPLATE.format(document=document_text, summary=summary_text)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    t0 = time.time()
    last_err = None
    for attempt in range(4):
        try:
            resp = requests.post(URL, headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            raw = resp.json()
            elapsed = time.time() - t0
            content = raw["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(content)
                parse_ok = isinstance(parsed, dict)
                if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                    parsed, parse_ok = parsed[0], True
            except Exception:
                parsed, parse_ok = {"_raw": content}, False
            usage = raw.get("usage", {})
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            return {
                "doc_id": doc["id"],
                "il_tur_doc_tokens": doc["num_doc_tokens"],
                "il_tur_summ_tokens": doc["num_summ_tokens"],
                "prompt_token_count": usage.get("prompt_tokens"),
                "candidates_token_count": usage.get("completion_tokens"),
                "reasoning_tokens": reasoning_tokens,
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


def main() -> None:
    summ = load_from_disk("data/raw/summ")
    train = summ["train"]
    idxs = random.Random(SEED).sample(range(len(train)), N_SAMPLES)

    out_dir = Path("data/pilot") / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    already_done = {int(p.stem) for p in out_dir.glob("*.json") if p.stem.isdigit()}
    docs = [train[i] for i in idxs]
    results = []
    to_generate = []
    for doc in docs:
        if doc["id"] in already_done:
            with open(out_dir / f"{doc['id']}.json", encoding="utf-8") as f:
                results.append(json.load(f))
        else:
            to_generate.append(doc)

    print(f"Reusing {len(results)}, generating {len(to_generate)} new "
          f"({MAX_WORKERS} parallel workers)...")

    token = get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(generate_one, headers, doc): doc for doc in to_generate}
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
        total_reasoning = sum(r["reasoning_tokens"] for r in ok)
        print(f"\n=== AGGREGATE: {len(ok)}/{len(results)} succeeded, {n_parsed} valid JSON dict ===")
        print(f"Input ratio: {in_ratio:.3f}  Output ratio: {out_ratio:.3f}")
        print(f"Total reasoning tokens: {total_reasoning} (should be ~0, non-reasoning model)")

    with open(out_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} total results to {out_dir}/")


if __name__ == "__main__":
    main()
