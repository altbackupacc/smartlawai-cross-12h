"""Batched HHEM claim-level scoring for the FULL M1 Path 2 corpus
(data/processed/section_pairs_raw.jsonl, ~31,621 pairs, ~212,000 claims).

Unlike the pilot's hhem_score_vertex.py (one model.predict() call per claim,
~212,000 tiny GPU calls, est. ~7-8hrs), this batches many (premise_chunk,
claim) work items into large forward passes (BATCH_SIZE at a time) so the L4
actually does parallel work instead of sitting mostly idle between calls.

Runs as a Vertex AI Custom Job on a single L4 -- confirmed sufficient for
this workload (HHEM is a small model; batching, not more GPUs, was the fix).

Input:  data/processed/section_pairs_raw.jsonl (from run_path2_production.py)
Output (both to GCS, mirrored to the pattern used throughout the pilot):
  - section_pairs_scored.jsonl: every pair + hhem claim scores
  - section_pairs_accepted.jsonl: only pairs with pass_rate >= ACCEPT_THRESHOLD
  - summary.json: acceptance rate + aggregate stats for data/PROVENANCE.md
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
SHARD_NAME = sys.argv[1] if len(sys.argv) > 1 else "raw"  # e.g. "l4_1", "a100"
INPUT_BLOB = f"shards/section_pairs_{SHARD_NAME}.jsonl" if SHARD_NAME != "raw" else "section_pairs_raw.jsonl"
SCORED_BLOB = f"shards/section_pairs_scored_{SHARD_NAME}.jsonl"
ACCEPTED_BLOB = f"shards/section_pairs_accepted_{SHARD_NAME}.jsonl"
SUMMARY_BLOB = f"shards/hhem_summary_{SHARD_NAME}.json"

MODEL_ID = "vectara/hallucination_evaluation_model"
PREMISE_CHUNK_WORDS = 300
PREMISE_CHUNK_OVERLAP = 50
ACCEPT_THRESHOLD = 0.8
BATCH_SIZE = 32  # was 128 -- caused an 18.96GiB OOM on A100 when a batch happened
                 # to contain several unusually long premise chunks (padding blows
                 # up quadratically with sequence length in attention); 32 keeps
                 # worst-case padding memory well within even a single L4's 24GB
CHECKPOINT_EVERY = 200  # batches -- upload partial results periodically so a
                        # crash near the end doesn't lose all progress like the
                        # A100 shard's OOM did at 88.8% with zero output saved


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
    raw_text = bucket.blob(INPUT_BLOB).download_as_text()
    pairs = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    print(f"Loaded {len(pairs)} section pairs", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device)

    # Flatten every (premise_chunk, claim) work item across the WHOLE corpus,
    # tagged with (pair_idx, claim_idx) so scores can be reassembled after
    # batched scoring -- this is the actual fix vs. the unbatched pilot script.
    work_items = []  # (pair_idx, claim_idx, premise_chunk, claim_text)
    pair_claims: list[list[str]] = []
    for pair_idx, pair in enumerate(pairs):
        claims = split_claims(pair["generated_text"])
        pair_claims.append(claims)
        if not claims:
            continue
        premise_chunks = chunk_premise(pair["source_headnote"])
        for claim_idx, claim in enumerate(claims):
            for chunk in premise_chunks:
                work_items.append((pair_idx, claim_idx, chunk, claim))

    print(f"Total (premise_chunk, claim) work items: {len(work_items)}, "
          f"batching at {BATCH_SIZE}...", flush=True)

    checkpoint_blob = bucket.blob(f"shards/checkpoint_{SHARD_NAME}.json")
    claim_max_scores: dict[int, dict[int, float]] = {}
    start_batch = 0
    if checkpoint_blob.exists():
        ckpt = json.loads(checkpoint_blob.download_as_text())
        claim_max_scores = {int(k): {int(ck): cv for ck, cv in v.items()}
                             for k, v in ckpt["claim_max_scores"].items()}
        start_batch = ckpt["next_batch"]
        print(f"Resuming from checkpoint: batch {start_batch}", flush=True)

    def score_batch_safe(batch):
        """Score a batch; on OOM, retry as smaller sub-batches instead of
        crashing the whole job (the A100 failure: one oversized batch lost
        90%+ of completed work with no fallback)."""
        try:
            pairs_for_model = [(w[2], w[3]) for w in batch]
            return model.predict(pairs_for_model)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            print(f"  OOM on batch of {len(batch)}, splitting and retrying", flush=True)
            return list(score_batch_safe(batch[:mid])) + list(score_batch_safe(batch[mid:]))

    n_batches = (len(work_items) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(start_batch, n_batches):
        batch = work_items[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        scores = score_batch_safe(batch)
        for (pair_idx, claim_idx, _, _), score in zip(batch, scores):
            d = claim_max_scores.setdefault(pair_idx, {})
            d[claim_idx] = max(d.get(claim_idx, -1.0), float(score))
        if (b + 1) % 50 == 0 or b == n_batches - 1:
            print(f"  batch {b+1}/{n_batches} ({(b+1)*BATCH_SIZE} items)", flush=True)
        if (b + 1) % CHECKPOINT_EVERY == 0:
            checkpoint_blob.upload_from_string(
                json.dumps({"claim_max_scores": claim_max_scores, "next_batch": b + 1}),
                content_type="application/json")
            print(f"  checkpointed at batch {b+1}", flush=True)

    scored_pairs = []
    accepted_pairs = []
    all_claim_scores = []
    for pair_idx, pair in enumerate(pairs):
        claims = pair_claims[pair_idx]
        scores_by_claim = claim_max_scores.get(pair_idx, {})
        claim_scores = [scores_by_claim.get(i) for i in range(len(claims))]
        claim_scores = [s for s in claim_scores if s is not None]
        if claim_scores:
            pass_rate = sum(1 for s in claim_scores if s >= 0.5) / len(claim_scores)
            mean_score = sum(claim_scores) / len(claim_scores)
        else:
            pass_rate, mean_score = None, None
        record = {**pair, "hhem_claim_scores": claim_scores,
                  "hhem_mean_score": mean_score, "hhem_pass_rate": pass_rate,
                  "hhem_accepted": pass_rate is not None and pass_rate >= ACCEPT_THRESHOLD}
        scored_pairs.append(record)
        all_claim_scores.extend(claim_scores)
        if record["hhem_accepted"]:
            accepted_pairs.append(record)

    n_below = sum(1 for s in all_claim_scores if s < 0.5)
    summary = {
        "total_pairs": len(pairs),
        "accepted_pairs": len(accepted_pairs),
        "acceptance_rate_pct": len(accepted_pairs) / len(pairs) * 100 if pairs else 0,
        "total_claims": len(all_claim_scores),
        "mean_claim_score": sum(all_claim_scores) / len(all_claim_scores) if all_claim_scores else None,
        "hallucination_rate_pct": n_below / len(all_claim_scores) * 100 if all_claim_scores else None,
        "accept_threshold": ACCEPT_THRESHOLD,
    }
    print("\n=== FULL CORPUS HHEM SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)

    bucket.blob(SCORED_BLOB).upload_from_string(
        "\n".join(json.dumps(r) for r in scored_pairs), content_type="application/jsonl")
    bucket.blob(ACCEPTED_BLOB).upload_from_string(
        "\n".join(json.dumps(r) for r in accepted_pairs), content_type="application/jsonl")
    bucket.blob(SUMMARY_BLOB).upload_from_string(
        json.dumps(summary, indent=2), content_type="application/json")
    print(f"\nUploaded scored/accepted/summary to gs://{BUCKET}/", flush=True)


if __name__ == "__main__":
    main()
