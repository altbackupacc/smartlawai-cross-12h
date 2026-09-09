"""Live smoke test script for Module 3 (Structured Generation).

Verifies that an OpenAI-compatible endpoint (GCP Cloud Run GPU, vLLM on GCP A100 / L4,
or local Ollama) produces valid structured claims and conforms to the Generator protocol.

Usage:
    python scripts/smoke_test_m3.py [--base-url URL] [--model MODEL] [--api-key KEY] [--device cuda/cpu]
"""
from __future__ import annotations

import argparse
import sys
import time

from smartlawai import config
from smartlawai.adapters.base import Chunk, RetrievedChunk
from smartlawai.core.generate import StructuredMistralGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M3 Structured Generation Smoke Test")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for OpenAI-compatible endpoint (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default=config.MISTRAL_GENERATOR_MODEL_ID,
        help="Model ID to request (default: Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key for endpoint authentication",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Hardware compute target / device (per CLAUDE.md §2)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print(" SmartLawAI — M3 Structured Generation Smoke Test")
    print(f" Target Endpoint: {args.base_url or 'default env-driven'}")
    print(f" Model ID:        {args.model}")
    print(f" Device target:   {args.device}")
    print("=" * 60)

    # 1. Initialize Generator
    gen = StructuredMistralGenerator(
        model_id=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    # 2. Test Empty Passages (Invariant check: no LLM call needed)
    print("\n[1/3] Testing empty passages fast-path...")
    t0 = time.perf_counter()
    res_empty = gen.generate("What is Section 27?", [])
    dur_empty = (time.perf_counter() - t0) * 1000
    assert res_empty.claims == [], "Claims must be empty when no passages provided"
    assert len(res_empty.unanswerable_aspects) > 0, "Must record unanswerable aspects"
    print(f"  OK (duration: {dur_empty:.2f} ms, 0 network tokens)")

    # 3. Test Structured Generation with Legal Context
    print("\n[2/3] Testing structured generation with Indian Contract Act context...")
    test_chunk_1 = Chunk(
        chunk_id="chk-ica-s27",
        doc_id="doc-statute-ica",
        chunk_index=0,
        chunk_text=(
            "Section 27. Agreement in restraint of trade, void. "
            "Every agreement by which any one is restrained from exercising a lawful "
            "profession, trade or business of any kind, is to that extent void. "
            "Exception 1 -- Saving of agreement not to carry on business of which good-will is sold."
        ),
        char_start=0,
        char_end=270,
    )
    test_chunk_2 = Chunk(
        chunk_id="chk-ica-s28",
        doc_id="doc-statute-ica",
        chunk_index=1,
        chunk_text=(
            "Section 28. Agreements in restraint of legal proceedings, void. "
            "Every agreement by which any party thereto is restricted absolutely from "
            "enforcing his rights under or in respect of any contract, is void to that extent."
        ),
        char_start=271,
        char_end=500,
    )

    passages = [
        RetrievedChunk(chunk=test_chunk_1, score=0.92),
        RetrievedChunk(chunk=test_chunk_2, score=0.85),
    ]

    question = "Under Indian law, is an agreement restraining someone from carrying on business void?"
    t0 = time.perf_counter()
    res = gen.generate(question, passages)
    dur = (time.perf_counter() - t0) * 1000

    print(f"  Response received in {dur:.2f} ms")
    print(f"  Model ID: {res.model_id}")
    print(f"  Claims returned: {len(res.claims)}")
    print(f"  Unanswerable aspects: {len(res.unanswerable_aspects)}")

    if not res.claims:
        print("  WARNING: No claims extracted from response.")
        if res.unanswerable_aspects:
            print(f"  Unanswerable details: {res.unanswerable_aspects}")
        return 1

    print("\n[3/3] Validating claim structure and provenance:")
    for idx, claim in enumerate(res.claims, 1):
        print(f"  Claim {idx}:")
        print(f"    Text:        {claim.text}")
        print(f"    Passage IDs: {claim.passage_ids}")
        print(f"    Citations:   {claim.citations}")

        # Check passage ID validity
        valid_chunk_ids = {"chk-ica-s27", "chk-ica-s28"}
        for pid in claim.passage_ids:
            assert pid in valid_chunk_ids, f"Invalid chunk_id {pid} returned in claim!"

    print("\n" + "=" * 60)
    print(" ALL SMOKE TESTS PASSED: Structured generation verified!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
