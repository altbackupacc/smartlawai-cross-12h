"""Build the Vertex AI Batch Prediction input JSONL for Path 2 generation
(the real M1 run, not the pilot). One request per summ document, all splits
(train+test) -- dedup/split happens afterward in dedup_split.py, per the
original M1 pipeline order.

Each line matches Vertex's batch prediction request format for Gemini:
{"request": {"contents": [...], "generationConfig": {...}}}
Includes a custom "doc_id" key alongside "request" (ignored by Vertex, passed
through to the corresponding output record) so results can be matched back
to source documents.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk

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


def build_request(doc_id: str, document_text: str, summary_text: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(document=document_text, summary=summary_text)
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        },
        "doc_id": doc_id,
    }


def main(limit: int | None, out_path: Path) -> None:
    summ = load_from_disk("data/raw/summ")
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for split_name in ["train", "test"]:
            for row in summ[split_name]:
                if limit is not None and n_written >= limit:
                    print(f"Wrote {n_written} requests to {out_path} (limited)")
                    return
                document_text = " ".join(row["document"])
                summary_text = " ".join(row["summary"])
                req = build_request(row["id"], document_text, summary_text)
                f.write(json.dumps(req) + "\n")
                n_written += 1
    print(f"Wrote {n_written} requests to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only write the first N requests (smoke test).")
    parser.add_argument("--out", type=Path, default=Path("data/processed/batch_input.jsonl"))
    args = parser.parse_args()
    main(args.limit, args.out)
