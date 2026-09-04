"""HHEM claim-level scoring + filtering for Path 2 section pairs (M1).

Runs after build_pairs.py's Path 2 generation, before dedup_split.py. Scores
every generated section against the ORIGINAL headnote it was derived from
(claim-level, per RESEARCH.md's own metric definition), and only pairs
meeting ACCEPT_THRESHOLD proceed into the corpus -- turning RESEARCH.md T4's
"human-validate 200, report acceptance rate" into a full automated filter
over every pair, with the 200-item human check now validating the FILTER's
own judgment rather than being the only quality check.

Designed to run as a Vertex AI Custom Job on L4 (see data/pilot/
hhem_score_vertex.py for the proven job-submission pattern this generalizes)
-- HHEM at this scale needs GPU, not local CPU/quick-check.

Input:  JSONL, one section pair per line:
        {"doc_id": ..., "section": ..., "source_headnote": "<original summ text>",
         "generated_text": "<Path-2 model output for this section>", ...other fields}
Output: same JSONL + hhem fields, plus a filtered *_accepted.jsonl containing
        only pairs that passed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MODEL_ID = "vectara/hallucination_evaluation_model"
PREMISE_CHUNK_WORDS = 300
PREMISE_CHUNK_OVERLAP = 50

# A pair is ACCEPTED only if at least this fraction of its individual claims
# score >=0.5 (HHEM's own "likely faithful" cutoff). Section-level (not claim-level)
# accept/reject, because dropping only the bad sentences out of an otherwise-coherent
# section leaves broken text -- whole-pair accept/reject keeps every kept pair coherent.
# 0.8 is a starting point, not yet calibrated against human judgment -- M5's job is to
# calibrate real thresholds against labelled data (PLAN.md M5.6's precedent for the OOS
# threshold); revisit this once the 200-item human validation sample exists to check it
# against.
ACCEPT_THRESHOLD = 0.8


def split_claims(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_premise(text: str) -> list[str]:
    words = text.split()
    if len(words) <= PREMISE_CHUNK_WORDS:
        return [text]
    chunks = []
    step = PREMISE_CHUNK_WORDS - PREMISE_CHUNK_OVERLAP
    for start in range(0, len(words), step):
        chunk = words[start:start + PREMISE_CHUNK_WORDS]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + PREMISE_CHUNK_WORDS >= len(words):
            break
    return chunks


def load_scorer():
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected. This is fine for a handful of pairs (pilot-scale) "
              "but will be slow at full-corpus scale (~35k pairs) -- run as a Vertex AI "
              "Custom Job on L4 for the real run, per data/pilot/hhem_score_vertex.py's "
              "proven pattern.", file=sys.stderr)
    return model.to(device), device


def score_pair(model, headnote: str, generated_text: str) -> dict:
    premise_chunks = chunk_premise(headnote)
    claims = split_claims(generated_text)
    claim_scores = []
    for claim in claims:
        batch = [(c, claim) for c in premise_chunks]
        per_chunk = [float(s) for s in model.predict(batch)]
        claim_scores.append(max(per_chunk))
    if not claim_scores:
        return {"n_claims": 0, "claim_scores": [], "mean_score": None,
                "pass_rate": None, "accepted": False}
    pass_rate = sum(1 for s in claim_scores if s >= 0.5) / len(claim_scores)
    return {
        "n_claims": len(claims),
        "claim_scores": claim_scores,
        "mean_score": sum(claim_scores) / len(claim_scores),
        "pass_rate": pass_rate,
        "accepted": pass_rate >= ACCEPT_THRESHOLD,
    }


def score_and_filter(in_path: Path, out_path: Path, accepted_path: Path) -> None:
    model, device = load_scorer()

    scored, accepted = [], []
    with open(in_path, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    print(f"Scoring {len(pairs)} pairs on {device}...")
    for i, pair in enumerate(pairs, 1):
        result = score_pair(model, pair["source_headnote"], pair["generated_text"])
        record = {**pair, "hhem": result}
        scored.append(record)
        if result["accepted"]:
            accepted.append(record)
        if i % 50 == 0 or i == len(pairs):
            print(f"  [{i}/{len(pairs)}] running acceptance rate: "
                  f"{len(accepted)}/{i} = {len(accepted)/i*100:.1f}%")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    with open(accepted_path, "w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r) + "\n")

    rate = len(accepted) / len(pairs) * 100 if pairs else 0.0
    print(f"\n=== ACCEPTANCE SUMMARY ===")
    print(f"Total pairs scored: {len(pairs)}")
    print(f"Accepted (pass_rate >= {ACCEPT_THRESHOLD}): {len(accepted)} ({rate:.1f}%)")
    print(f"Rejected: {len(pairs) - len(accepted)} ({100 - rate:.1f}%)")
    print(f"\nThis acceptance rate belongs in data/PROVENANCE.md (RESEARCH.md T4) -- "
          f"report it plainly regardless of what it turns out to be.")
    print(f"Scored (all): {out_path}")
    print(f"Accepted only: {accepted_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", dest="out_path", required=True, type=Path)
    parser.add_argument("--accepted", dest="accepted_path", required=True, type=Path)
    args = parser.parse_args()
    score_and_filter(args.in_path, args.out_path, args.accepted_path)
