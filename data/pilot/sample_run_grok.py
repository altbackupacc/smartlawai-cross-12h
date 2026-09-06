"""Sample-document trial run against Grok 4.20 Reasoning via Vertex AI.
Uses the OpenAI-compatible chat/completions endpoint (global location --
xAI models on Vertex are global-only, same as Gemini 3.x), since Grok is
"OpenMaaS" and can't be called via PredictionServiceClient.raw_predict.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import random

import google.auth
import google.auth.transport.requests
import requests
from datasets import load_from_disk

MODEL = "xai/grok-4.20-reasoning"
OUT_DIR_NAME = "grok-4.20-reasoning"
URL = "https://aiplatform.googleapis.com/v1/projects/smartlawai-1/locations/global/endpoints/openapi/chat/completions"
N_SAMPLES = 10
SEED = 42

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


def main() -> None:
    summ = load_from_disk("data/raw/summ")
    train = summ["train"]
    rng = random.Random(SEED)
    idxs = rng.sample(range(len(train)), N_SAMPLES)

    out_dir = Path("data/pilot") / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume mode: skip docs already scored successfully in a prior run (the first
    # attempt hit 429s on 3/10 with no pacing -- retry only the missing ones).
    already_done = {int(p.stem) for p in out_dir.glob("*.json") if p.stem.isdigit()}

    token = get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    results = []
    for n, idx in enumerate(idxs, 1):
        doc = train[idx]
        if doc["id"] in already_done:
            print(f"[{n}/{N_SAMPLES}] doc_id={doc['id']} -- already done, skipping")
            with open(out_dir / f"{doc['id']}.json", encoding="utf-8") as f:
                results.append(json.load(f))
            continue
        document_text = " ".join(doc["document"])
        summary_text = " ".join(doc["summary"])
        prompt = PROMPT_TEMPLATE.format(document=document_text, summary=summary_text)

        print(f"[{n}/{N_SAMPLES}] doc_id={doc['id']} "
              f"(doc_tokens={doc['num_doc_tokens']}, summ_tokens={doc['num_summ_tokens']})")

        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        t0 = time.time()
        raw, err = None, None
        for attempt in range(4):
            try:
                resp = requests.post(URL, headers=headers, json=body, timeout=180)
                resp.raise_for_status()
                raw = resp.json()
                break
            except Exception as e:
                err = e
                if "429" in str(e) and attempt < 3:
                    wait = 15 * (attempt + 1)
                    print(f"  429, backing off {wait}s (attempt {attempt+1}/4)")
                    time.sleep(wait)
                else:
                    break
        if raw is None:
            print(f"  ERROR: {err}")
            results.append({"doc_id": doc["id"], "error": str(err)})
            continue
        elapsed = time.time() - t0

        content = raw["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            parse_ok = isinstance(parsed, dict)
        except Exception:
            parsed = {"_raw": content}
            parse_ok = False

        usage = raw.get("usage", {})
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        record = {
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
        results.append(record)
        with open(out_dir / f"{doc['id']}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        print(f"  input={usage.get('prompt_tokens')} output={usage.get('completion_tokens')} "
              f"reasoning={reasoning_tokens} parse_ok={parse_ok} elapsed={elapsed:.1f}s")
        time.sleep(10)  # pacing to avoid re-tripping the rate limit

    ok = [r for r in results if "error" not in r]
    if ok:
        in_ratio = sum(r["prompt_token_count"] for r in ok) / sum(
            r["il_tur_doc_tokens"] + r["il_tur_summ_tokens"] for r in ok)
        out_ratio = sum(r["candidates_token_count"] for r in ok) / sum(
            r["il_tur_summ_tokens"] for r in ok)
        total_reasoning = sum(r["reasoning_tokens"] for r in ok)
        n_parsed = sum(1 for r in ok if r["parse_ok"])
        print("\n=== AGGREGATE ===")
        print(f"Samples: {len(ok)}/{N_SAMPLES} succeeded, {n_parsed}/{len(ok)} valid JSON dict")
        print(f"Input tokens / (doc+summ IL-TUR tokens) ratio: {in_ratio:.3f}")
        print(f"Output tokens / summ IL-TUR tokens ratio: {out_ratio:.3f}")
        print(f"Total reasoning tokens burned: {total_reasoning:,} "
              f"(avg {total_reasoning/len(ok):.0f}/doc)")

    with open(out_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_dir}/")


if __name__ == "__main__":
    main()
