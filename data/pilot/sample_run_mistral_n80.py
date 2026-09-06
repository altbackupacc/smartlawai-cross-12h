"""Bigger-sample (n=80) run against Mistral Medium 3. Sequential, not parallel --
Mistral's actual quota is 9 req/min regardless of thread count, so concurrency
would just cause more 429 retries, not go faster. Reuses the original 10 docs'
already-saved results (data/pilot/mistral-medium-3/) and generates 70 new ones
(same seed=43 extra-doc selection as the Gemini n=80 run, for a like-for-like
comparison on an overlapping-as-possible set).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from google.api import httpbody_pb2
from google.cloud import aiplatform_v1
from datasets import load_from_disk

MODEL = "mistral-medium-3"
PUBLISHER = "mistralai"
REGION = "us-central1"
ORIGINAL_N = 10
TARGET_N = 80
ORIGINAL_SEED = 42
EXTRA_SEED = 43

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


def call_mistral(client, endpoint: str, prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    http_body = httpbody_pb2.HttpBody(content_type="application/json",
                                       data=json.dumps(body).encode("utf-8"))
    resp = client.raw_predict(endpoint=endpoint, http_body=http_body)
    return json.loads(resp.data)


def main() -> None:
    summ = load_from_disk("data/raw/summ")
    train = summ["train"]

    original_idxs = random.Random(ORIGINAL_SEED).sample(range(len(train)), ORIGINAL_N)
    original_ids = {train[i]["id"] for i in original_idxs}
    pool = [i for i in range(len(train)) if train[i]["id"] not in original_ids]
    extra_idxs = random.Random(EXTRA_SEED).sample(pool, TARGET_N - ORIGINAL_N)

    out_dir = Path("data/pilot/mistral-medium-3-n80")
    out_dir.mkdir(parents=True, exist_ok=True)
    reuse_dir = Path("data/pilot/mistral-medium-3")

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
    for i in extra_idxs:
        doc = train[i]
        existing = out_dir / f"{doc['id']}.json"
        if existing.exists():
            with open(existing, encoding="utf-8") as f:
                results.append(json.load(f))
        else:
            to_generate.append(doc)

    print(f"Reusing {len(results)} existing results, generating {len(to_generate)} new "
          f"(sequential, ~7s pacing for the 9 req/min quota)...")

    client = aiplatform_v1.PredictionServiceClient(
        client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    )
    endpoint = f"projects/smartlawai-1/locations/{REGION}/publishers/{PUBLISHER}/models/{MODEL}"

    for n, doc in enumerate(to_generate, 1):
        document_text = " ".join(doc["document"])
        summary_text = " ".join(doc["summary"])
        prompt = PROMPT_TEMPLATE.format(document=document_text, summary=summary_text)

        t0 = time.time()
        raw, err = None, None
        for attempt in range(4):
            try:
                raw = call_mistral(client, endpoint, prompt)
                break
            except Exception as e:
                err = e
                if "429" in str(e) and attempt < 3:
                    time.sleep(15 * (attempt + 1))
                else:
                    break
        elapsed = time.time() - t0

        if raw is None:
            print(f"[{n}/{len(to_generate)}] doc_id={doc['id']} -> ERROR: {err}")
            results.append({"doc_id": doc["id"], "error": str(err)})
            continue

        content = raw["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            parse_ok = isinstance(parsed, dict)
        except Exception:
            parsed, parse_ok = {"_raw": content}, False

        usage = raw.get("usage", {})
        record = {
            "doc_id": doc["id"],
            "il_tur_doc_tokens": doc["num_doc_tokens"],
            "il_tur_summ_tokens": doc["num_summ_tokens"],
            "prompt_token_count": usage.get("prompt_tokens"),
            "candidates_token_count": usage.get("completion_tokens"),
            "elapsed_s": round(elapsed, 1),
            "parse_ok": parse_ok,
            "sections": parsed,
        }
        results.append(record)
        with open(out_dir / f"{doc['id']}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        print(f"[{n}/{len(to_generate)}] doc_id={doc['id']} -> OK parse_ok={parse_ok}")
        time.sleep(7)

    ok = [r for r in results if "error" not in r]
    n_parsed = sum(1 for r in ok if r["parse_ok"])
    print(f"\n=== AGGREGATE: {len(ok)}/{len(results)} succeeded, {n_parsed}/{len(ok)} valid JSON ===")

    with open(out_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} total results to {out_dir}/")


if __name__ == "__main__":
    main()
