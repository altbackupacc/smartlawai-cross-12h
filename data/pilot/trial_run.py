"""One-document trial run against Gemini 2.5 Flash-Lite via Vertex AI.
Scratch script for the E.3 model pilot -- not the real build_pairs.py Path 2
implementation. Confirms the prompt design + structured output work before
scaling to the full 5-model x 15-20 doc pilot.
"""
from __future__ import annotations

import json

from datasets import load_from_disk
from google import genai
from google.genai import types

SECTIONS = ["facts", "statute", "argument", "analysis", "judgement"]  # IN-Ext taxonomy

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
    doc = summ["train"][0]
    document_text = " ".join(doc["document"])
    summary_text = " ".join(doc["summary"])

    prompt = PROMPT_TEMPLATE.format(document=document_text, summary=summary_text)

    client = genai.Client(vertexai=True, project="smartlawai-1", location="us-central1")
    resp = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

    print("=== RAW RESPONSE ===")
    print(resp.text)

    result = json.loads(resp.text)
    result["_doc_id"] = doc["id"]
    result["_model"] = "gemini-2.5-flash-lite"
    result["_input_tokens_est"] = doc["num_doc_tokens"] + doc["num_summ_tokens"]

    with open("data/pilot/trial_gemini-2.5-flash-lite.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== USAGE ===")
    print(resp.usage_metadata)
    print("\nSaved to data/pilot/trial_gemini-2.5-flash-lite.json")


if __name__ == "__main__":
    main()
