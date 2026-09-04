"""Score the Path-2 pilot outputs' faithfulness with HHEM-2.1 (Vectara's
hallucination evaluation model), comparing each generated section against the
ORIGINAL headnote it was supposed to be derived from -- exactly what the
Path-2 prompt asked the model to stay faithful to.

Pilot/scratch script for the E.3 model pilot. Not the vendored, trust_remote_
code-free HHEM integration M5 will eventually build for the serving gate
(PLAN.md M5.2's "vendor the HHEM model class" requirement) -- this uses the
standard HF interface for a quick, honest faithfulness read.
"""
from __future__ import annotations

import json
from pathlib import Path

MODEL_ID = "vectara/hallucination_evaluation_model"


PREMISE_CHUNK_WORDS = 300  # conservative vs HHEM's 512-token limit (~1.3-1.5 tok/word
                           # for legal English) -- leaves headroom for the hypothesis too
PREMISE_CHUNK_OVERLAP = 50  # words, so a fact split across a chunk boundary isn't lost


def load_scorer():
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    return model


def split_claims(text: str) -> list[str]:
    """Split a generated section into individual sentences ("claims"), matching
    RESEARCH.md's own metric definition ("Faithfulness | HHEM-2.1, claim-level") --
    scoring a whole multi-sentence section as one hypothesis conflates claims that
    each trace to a different, separate part of the source premise."""
    import re
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


def score_pilot(pilot_dir: Path, out_path: Path) -> None:
    model = load_scorer()

    summary_path = pilot_dir / "_summary.json"
    with open(summary_path, encoding="utf-8") as f:
        results = json.load(f)

    pairs = []  # (doc_id, section_name, premise=headnote, hypothesis=generated_section)
    from datasets import load_from_disk
    summ = load_from_disk("data/raw/summ")
    train = summ["train"]
    id_to_summary = {}
    for row in train:
        id_to_summary[row["id"]] = " ".join(row["summary"])

    for r in results:
        if "error" in r or not r.get("parse_ok"):
            continue
        doc_id = r["doc_id"]
        headnote = id_to_summary.get(doc_id, "")
        for section, text in r["sections"].items():
            if not text:
                continue
            pairs.append((doc_id, section, headnote, text))

    print(f"Scoring {len(pairs)} sections at CLAIM level (split into sentences), each "
          f"claim vs premise chunked over {PREMISE_CHUNK_WORDS} words, max score per "
          f"claim across chunks -- matches RESEARCH.md's claim-level metric definition...")
    scored = []
    for i, (doc_id, section, headnote, text) in enumerate(pairs, 1):
        premise_chunks = chunk_premise(headnote)
        claims = split_claims(text)
        claim_scores = []
        for claim in claims:
            per_chunk = [float(s) for s in model.predict([(c, claim) for c in premise_chunks])]
            claim_scores.append(max(per_chunk))
        section_score = sum(claim_scores) / len(claim_scores) if claim_scores else None
        pass_rate = (sum(1 for s in claim_scores if s >= 0.5) / len(claim_scores)
                     if claim_scores else None)
        scored.append({
            "doc_id": doc_id,
            "section": section,
            "n_claims": len(claims),
            "claim_scores": claim_scores,
            "mean_claim_score": section_score,
            "claim_pass_rate": pass_rate,
            "text": text,
        })
        mean_str = f"{section_score:.3f}" if section_score is not None else "n/a"
        print(f"  [{i}/{len(pairs)}] doc={doc_id} section={section} n_claims={len(claims)} "
              f"mean={mean_str} pass_rate={pass_rate}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2)

    # Claim-level aggregate: pool every individual claim's score across all sections,
    # not the per-section means -- this is the real "% of claims not entailed" number
    # CLAUDE.md sec5 defines as THE hallucination-rate metric for this project.
    all_claim_scores = [s for rec in scored for s in rec["claim_scores"]]
    print(f"\n=== CLAIM-LEVEL HHEM SCORES (n={len(all_claim_scores)} claims, "
          f"{len(scored)} sections) ===")
    print(f"Mean claim score: {sum(all_claim_scores)/len(all_claim_scores):.4f}")
    print(f"Min:  {min(all_claim_scores):.4f}")
    print(f"Max:  {max(all_claim_scores):.4f}")
    n_below = sum(1 for s in all_claim_scores if s < 0.5)
    print(f"Claims below 0.5 (= not entailed -> hallucination rate): "
          f"{n_below}/{len(all_claim_scores)} = {n_below/len(all_claim_scores)*100:.1f}%")

    # Worst individual claims, for spot-checking real vs measurement-artifact failures
    flat = [(rec["doc_id"], rec["section"], claim, score)
            for rec in scored for claim, score in zip(split_claims(rec["text"]), rec["claim_scores"])]
    print("\nWorst 5 individual claims:")
    for doc_id, section, claim, score in sorted(flat, key=lambda x: x[3])[:5]:
        print(f"  [{score:.3f}] doc={doc_id} section={section}: {claim[:150]}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    score_pilot(Path("data/pilot/gemini-2.5-flash-lite"),
                Path("data/pilot/gemini-2.5-flash-lite_hhem.json"))
