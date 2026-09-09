"""M2 Retrieval Benchmark Runner: Runs InLegalBERT + BM25 + CrossEncoder rerank
under scoped retrieval (I1). Works seamlessly on CPU, local GPU, or GCP L4/A100 instances."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import torch

from smartlawai import config
from smartlawai.adapters.factory import get_backend
from smartlawai.core.encoders import InLegalBERTEncoder
from smartlawai.core.rerank import Reranker
from smartlawai.pipeline import Pipeline
from smartlawai.scope import Scope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartLawAI M2 Retrieval Benchmark")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run encoder/reranker on ('cuda', 'cpu', 'auto')")
    parser.add_argument("--backend", type=str, default="local",
                        help="Storage backend ('local', 'gcloud')")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for InLegalBERT encoder")
    parser.add_argument("--top-k", type=int, default=config.RERANK_TOP_K,
                        help="Top-k chunks to return after reranking")
    return parser.parse_args()


def _ensure_sample_document(pipeline: Pipeline, owner_id: str = "anon") -> str:
    """Ingests a real Indian legal text if no documents are in the backend."""
    sample_text = (
        "Section 10. What agreements are contracts.\n"
        "All agreements are contracts if they are made by the free consent of parties "
        "competent to contract, for a lawful consideration and with a lawful object, and "
        "are not hereby expressly declared to be void. Nothing herein contained shall affect "
        "any law in force in India, and not hereby expressly repealed, by which any contract "
        "is required to be made in writing or in the presence of witnesses, or any law relating "
        "to the registration of documents.\n\n"
        "Section 27. Agreement in restraint of trade, void.\n"
        "Every agreement by which anyone is restrained from exercising a lawful profession, "
        "trade or business of any kind, is to that extent void. Exception 1 -- Saving of agreement "
        "not to carry on business of which good-will is sold. One who sells the good-will of a business "
        "may agree with the buyer to refrain from carrying on a similar business, within specified local "
        "limits, so long as the buyer, or any person deriving title to the good-will from him, carries "
        "on a like business therein, provided that such limits appear to the Court reasonable, regard "
        "being had to the nature of the business.\n\n"
        "Section 74. Compensation for breach of contract where penalty stipulated for.\n"
        "When a contract has been broken, if a sum is named in the contract as the amount to be paid "
        "in case of such breach, or if the contract contains any other stipulation by way of penalty, "
        "the party complaining of the breach is entitled, whether or not actual damage or loss is "
        "proved to have been caused thereby, to receive from the party who has broken the contract "
        "reasonable compensation not exceeding the amount so named or, as the case may be, the penalty "
        "stipulated for."
    )
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(sample_text)
        tmp_path = f.name

    try:
        doc_id, _ = pipeline.ingest(tmp_path, doc_type="STATUTE", source="indian_contract_act", owner_id=owner_id)
        return doc_id
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print(f"[*] SmartLawAI M2 Retrieval Benchmark (Hardware: {args.device})")
    print("=" * 60)

    if torch.cuda.is_available() and "cuda" in args.device:
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[+] CUDA Device: {device_name} (Compute {capability[0]}.{capability[1]})")
        print(f"[+] Total GPU VRAM: {total_mem_gb:.2f} GB")
        print(f"[+] bfloat16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("[*] Running on CPU")

    t0 = time.perf_counter()
    backend = get_backend(args.backend)
    encoder = InLegalBERTEncoder(device=args.device)
    reranker = Reranker(device=args.device)
    pipeline = Pipeline(backend=backend, encoder=encoder, reranker=reranker)
    init_time = time.perf_counter() - t0
    print(f"[+] Loaded InLegalBERT & CrossEncoder in {init_time:.2f}s")

    doc_ids = backend.list_doc_ids()
    if not doc_ids:
        print("[*] Backend empty. Ingesting benchmark legal document...")
        target_doc = _ensure_sample_document(pipeline, owner_id="anon")
    else:
        target_doc = doc_ids[0]

    all_chunks = backend.fetch_chunks(target_doc)
    print(f"[+] Document ID: {target_doc} ({len(all_chunks)} total chunks)")

    # Execute scoped retrieval query
    scope = Scope(doc_ids=(target_doc,), owner_id="anon")
    query = "Is an agreement in restraint of trade valid or void under Indian law?"

    print(f"[*] Running scoped retrieval for query: '{query}'")
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() and "cuda" in args.device else None
    t1 = time.perf_counter()
    ans_res = pipeline.ask(query, scope=scope)
    elapsed = time.perf_counter() - t1

    top_ids = ans_res.trace.retrieval.get("top_chunk_ids", [])
    candidate_ids = ans_res.trace.retrieval.get("candidate_chunk_ids", [])

    print(f"[+] Scoped retrieval latency: {elapsed * 1000:.2f} ms")
    if torch.cuda.is_available() and "cuda" in args.device:
        peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        print(f"[+] Peak GPU Memory during query: {peak_vram_mb:.1f} MB")

    print(f"[+] Scoped candidate chunks searched: {len(candidate_ids)}")
    print(f"[+] Resolved top parent chunk IDs ({len(top_ids)}): {top_ids}")

    # Check top chunk content
    by_id = {c.chunk_id: c for c in all_chunks}
    for i, cid in enumerate(top_ids[:3]):
        c = by_id.get(cid)
        if c:
            label = c.section_label or "No label"
            preview = c.chunk_text.replace("\n", " ")[:90]
            print(f"    Rank {i + 1} [{label} | {c.chunk_id}]: {preview}...")

    # Invariant I1 Verification
    in_scope_chunks = {c.chunk_id for c in backend.fetch_chunks_scoped(scope)}
    leakage = [cid for cid in top_ids if cid not in in_scope_chunks]
    if leakage:
        print(f"[!] CRITICAL: Invariant I1 VIOLATION! Chunks {leakage} are out of scope!")
        return 1

    print("[+] INVARIANT I1 CONFIRMED: 100% of retrieved chunks belong strictly to the scoped document.")
    print("=" * 60)
    print("[*] M2 RETRIEVAL BENCHMARK COMPLETED SUCCESSFULLY.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
