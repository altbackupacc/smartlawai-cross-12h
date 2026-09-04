"""Sample-document trial run against Gemini 2.5 Flash-Lite via Vertex AI.
Scratch script for the E.3 model pilot -- runs N sample docs through the
Path-2 segmentation prompt, saves each result, and reports real token-usage
ratios vs IL-TUR's own token counts (for a more accurate cost estimate than
the pre-generation --dry-run figure).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from datasets import load_from_disk
from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash-lite"
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


def main() -> None:
    summ = load_from_disk("data/raw/summ")
    train = summ["train"]
    rng = random.Random(SEED)
    idxs = rng.sample(range(len(train)), N_SAMPLES)

    out_dir = Path("data/pilot") / MODEL
    out_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(vertexai=True, project="smartlawai-1", location="us-central1")

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
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"doc_id": doc["id"], "error": str(e)})
            continue
        elapsed = time.time() - t0

        try:
            parsed = json.loads(resp.text)
            parse_ok = True
        except Exception as e:
            parsed = {"_raw": resp.text}
            parse_ok = False

        usage = resp.usage_metadata
        record = {
            "doc_id": doc["id"],
            "il_tur_doc_tokens": doc["num_doc_tokens"],
            "il_tur_summ_tokens": doc["num_summ_tokens"],
            "prompt_token_count": usage.prompt_token_count,
            "candidates_token_count": usage.candidates_token_count,
            "elapsed_s": round(elapsed, 1),
            "parse_ok": parse_ok,
            "sections": parsed,
        }
        results.append(record)
        with open(out_dir / f"{doc['id']}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        print(f"  input={usage.prompt_token_count} output={usage.candidates_token_count} "
              f"parse_ok={parse_ok} elapsed={elapsed:.1f}s")

    # Aggregate stats
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
