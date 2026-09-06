"""Sample-document trial run against Mistral Medium 3 via Vertex AI.
Same prompt/methodology as sample_run.py (Gemini), adapted for Mistral's
OpenAI-compatible MaaS response format via the aiplatform PredictionServiceClient
(the genai SDK is Gemini-only; Mistral/DeepSeek need raw_predict).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from datasets import load_from_disk
from google.api import httpbody_pb2
from google.cloud import aiplatform_v1

MODEL = "mistral-medium-3"
PUBLISHER = "mistralai"
REGION = "us-central1"
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


def call_mistral(client, endpoint: str, prompt: str) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    http_body = httpbody_pb2.HttpBody(
        content_type="application/json", data=json.dumps(body).encode("utf-8")
    )
    resp = client.raw_predict(endpoint=endpoint, http_body=http_body)
    return json.loads(resp.data)


def main() -> None:
    summ = load_from_disk("data/raw/summ")
    train = summ["train"]
    rng = random.Random(SEED)
    idxs = rng.sample(range(len(train)), N_SAMPLES)

    out_dir = Path("data/pilot") / MODEL
    out_dir.mkdir(parents=True, exist_ok=True)

    client = aiplatform_v1.PredictionServiceClient(
        client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    )
    endpoint = f"projects/smartlawai-1/locations/{REGION}/publishers/{PUBLISHER}/models/{MODEL}"

    results = []
    for n, idx in enumerate(idxs, 1):
        doc = train[idx]
        document_text = " ".join(doc["document"])
        summary_text = " ".join(doc["summary"])
        prompt = PROMPT_TEMPLATE.format(document=document_text, summary=summary_text)

        print(f"[{n}/{N_SAMPLES}] doc_id={doc['id']} "
              f"(doc_tokens={doc['num_doc_tokens']}, summ_tokens={doc['num_summ_tokens']})")

        t0 = time.time()
        try:
            raw = call_mistral(client, endpoint, prompt)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"doc_id": doc["id"], "error": str(e)})
            continue
        elapsed = time.time() - t0

        content = raw["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            parse_ok = True
        except Exception:
            parsed = {"_raw": content}
            parse_ok = False

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

        print(f"  input={usage.get('prompt_tokens')} output={usage.get('completion_tokens')} "
              f"parse_ok={parse_ok} elapsed={elapsed:.1f}s")
        time.sleep(8)  # 9 req/min quota = ~6.7s/req minimum; 8s keeps real margin

    ok = [r for r in results if "error" not in r]
    if ok:
        in_ratio = sum(r["prompt_token_count"] for r in ok) / sum(
            r["il_tur_doc_tokens"] + r["il_tur_summ_tokens"] for r in ok)
        out_ratio = sum(r["candidates_token_count"] for r in ok) / sum(
            r["il_tur_summ_tokens"] for r in ok)
        n_parsed = sum(1 for r in ok if r["parse_ok"])
        print("\n=== AGGREGATE ===")
        print(f"Samples: {len(ok)}/{N_SAMPLES} succeeded, {n_parsed}/{len(ok)} valid JSON")
        print(f"Input tokens / (doc+summ IL-TUR tokens) ratio: {in_ratio:.3f}")
        print(f"Output tokens / summ IL-TUR tokens ratio: {out_ratio:.3f}")

    with open(out_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_dir}/")


if __name__ == "__main__":
    main()
