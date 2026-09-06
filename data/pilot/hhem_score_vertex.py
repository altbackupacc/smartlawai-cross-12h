"""Standalone HHEM claim-level scorer for a Vertex AI Custom Job (L4 GPU).
Reads staging_for_vertex.json from GCS, scores on GPU if available, writes
results back to GCS. Self-contained -- installs its own deps at runtime since
this runs in a generic PyTorch training container, not this repo's venv.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        "transformers>=4.40,<4.50", "sentencepiece", "google-cloud-storage"])

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402
from google.cloud import storage  # noqa: E402

BUCKET = "smartlawai-1-m1-pilot"
MODEL_TAG = sys.argv[1] if len(sys.argv) > 1 else ""  # e.g. "gemini-3.8-flash"
INPUT_BLOB = f"staging_{MODEL_TAG}.json" if MODEL_TAG else "staging_for_vertex.json"
OUTPUT_BLOB = f"hhem_results_{MODEL_TAG}.json" if MODEL_TAG else "hhem_results.json"
MODEL_ID = "vectara/hallucination_evaluation_model"
PREMISE_CHUNK_WORDS = 300
PREMISE_CHUNK_OVERLAP = 50


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


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(INPUT_BLOB)
    staging = json.loads(blob.download_as_text())
    results = staging["results"]
    headnotes = staging["headnotes"]

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device)

    malformed_structure = 0
    pairs = []
    for r in results:
        if "error" in r or not r.get("parse_ok"):
            continue
        sections = r["sections"]
        # Some models (observed: Gemini 3.1 Pro Preview) occasionally wrap the answer
        # in a list containing one dict instead of returning the dict directly, even
        # with response_mime_type=json and an explicit schema -- count it as a real
        # structural-reliability data point, not silently paper over it, but still
        # recover the content rather than dropping the whole document.
        if isinstance(sections, list):
            malformed_structure += 1
            sections = sections[0] if len(sections) == 1 and isinstance(sections[0], dict) else {}
        doc_id = r["doc_id"]
        headnote = headnotes.get(str(doc_id)) or headnotes.get(doc_id) or ""
        for section, text in sections.items():
            if section.startswith("_") or not text:
                continue
            pairs.append((doc_id, section, headnote, text))

    print(f"Scoring {len(pairs)} sections at claim level on {device}...", flush=True)
    scored = []
    for i, (doc_id, section, headnote, text) in enumerate(pairs, 1):
        premise_chunks = chunk_premise(headnote)
        claims = split_claims(text)
        claim_scores = []
        for claim in claims:
            batch = [(c, claim) for c in premise_chunks]
            per_chunk = [float(s) for s in model.predict(batch)]
            claim_scores.append(max(per_chunk))
        mean_score = sum(claim_scores) / len(claim_scores) if claim_scores else None
        scored.append({
            "doc_id": doc_id, "section": section, "n_claims": len(claims),
            "claim_scores": claim_scores, "mean_claim_score": mean_score, "text": text,
        })
        print(f"  [{i}/{len(pairs)}] doc={doc_id} section={section} "
              f"mean={mean_score if mean_score is not None else 'n/a'}", flush=True)

    all_claim_scores = [s for rec in scored for s in rec["claim_scores"]]
    n_below = sum(1 for s in all_claim_scores if s < 0.5)
    summary = {
        "device": device,
        "n_claims": len(all_claim_scores),
        "n_sections": len(scored),
        "n_malformed_structure": malformed_structure,
        "mean_claim_score": sum(all_claim_scores) / len(all_claim_scores) if all_claim_scores else None,
        "min": min(all_claim_scores) if all_claim_scores else None,
        "max": max(all_claim_scores) if all_claim_scores else None,
        "hallucination_rate_pct": (n_below / len(all_claim_scores) * 100) if all_claim_scores else None,
        "sections": scored,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "sections"}, indent=2), flush=True)

    out_blob = bucket.blob(OUTPUT_BLOB)
    out_blob.upload_from_string(json.dumps(summary, indent=2), content_type="application/json")
    print(f"Uploaded results to gs://{BUCKET}/{OUTPUT_BLOB}", flush=True)


if __name__ == "__main__":
    main()
