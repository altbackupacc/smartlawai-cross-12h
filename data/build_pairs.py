"""Materialize section pairs from summ (M1 step, PLAN.md / M1_ONBOARDING.md #6).

data/rr_summ_alignment_report.md decided: Path 2 primary, 0% RR<->SUMM overlap
(disjoint document universes), no Path 1 bonus subset. This script's Path 2
branch is the one actually used for this corpus; the Path 1 branch is kept for
completeness/reuse if a future corpus pull ever has real RR overlap.

--dry-run computes token volume and estimated dollar cost from summ's own
num_doc_tokens/num_summ_tokens fields -- no API calls, no cost. Per
M1_ONBOARDING.md #6, the real generation pass (--execute) must not run until
that estimate has been surfaced to the user and explicitly approved.
"""
from __future__ import annotations

import argparse
import json
import os

from datasets import load_from_disk

# Sections targeted per document -- PLAN.md's "7,100 documents x ~5 sections ~= 35,000
# section pairs" arithmetic. Segmentation + per-section summary generation are one LLM
# call's worth of output per document, not per section, so this only affects the output
# token estimate below, not the number of API calls.
TARGET_SECTIONS_PER_DOC = 5

# $ per million tokens (input, output). Spot-checked Aug 2026; re-check before a real run,
# prices move. Not exhaustive -- illustrative tiers to show the cost/quality tradeoff.
PRICING_PER_MILLION = {
    "gpt-5-mini-class (cheap)": (0.25, 2.00),
    "claude-haiku-4.5-class (cheap)": (1.00, 5.00),
    "claude-sonnet-5-class (capable)": (2.00, 10.00),
}


def estimate_cost(dry_run_only: bool = True) -> dict:
    summ = load_from_disk("data/raw/summ")

    total_docs = 0
    total_doc_tokens = 0
    total_summ_tokens = 0
    for split in summ.keys():
        total_docs += len(summ[split])
        total_doc_tokens += sum(summ[split]["num_doc_tokens"])
        total_summ_tokens += sum(summ[split]["num_summ_tokens"])

    # Input per doc: full judgment text + existing professional headnote (both needed
    # to segment the judgment into sections and align each section to part of the
    # headnote). Output per doc: TARGET_SECTIONS_PER_DOC section-level summaries;
    # estimated at 1.3x the existing single summary's token count to account for
    # restructuring into sections plus boundary/heading tokens -- a rough multiplier,
    # not a measured one; the real run's actual usage may differ.
    total_input_tokens = total_doc_tokens + total_summ_tokens
    total_output_tokens = int(total_summ_tokens * 1.3)

    costs = {}
    for name, (in_price, out_price) in PRICING_PER_MILLION.items():
        cost = (total_input_tokens / 1e6) * in_price + (total_output_tokens / 1e6) * out_price
        costs[name] = round(cost, 2)

    return {
        "total_docs": total_docs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost_usd_by_model_tier": costs,
        "expected_section_pairs": total_docs * TARGET_SECTIONS_PER_DOC,
    }


def print_estimate(est: dict) -> None:
    print(f"Documents to process (Path 2, full summ corpus): {est['total_docs']:,}")
    print(f"Estimated input tokens:  {est['total_input_tokens']:,}")
    print(f"Estimated output tokens: {est['total_output_tokens']:,}")
    print(f"Expected section pairs:  ~{est['expected_section_pairs']:,}")
    print("Estimated cost by model tier (spot-checked Aug 2026, re-verify before spending):")
    for name, cost in est["estimated_cost_usd_by_model_tier"].items():
        print(f"  {name}: ${cost}")
    print()
    print("This has zero line item in PLAN.md's COMPUTE BUDGET table.")
    print("Do NOT run --execute until this estimate has been surfaced to the user")
    print("and explicitly approved (M1_ONBOARDING.md #6).")


def run_path2_generation() -> None:
    raise NotImplementedError(
        "Path 2 real generation is gated behind explicit user cost approval "
        "(M1_ONBOARDING.md #6) -- not implemented until that approval is given."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the cost estimate only.")
    parser.add_argument("--execute", action="store_true",
                         help="Run the real Path 2 generation pass. Requires prior explicit "
                              "user approval of the --dry-run estimate.")
    args = parser.parse_args()

    if args.execute:
        run_path2_generation()
        return

    est = estimate_cost()
    print_estimate(est)
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/path2_cost_estimate.json", "w", encoding="utf-8") as f:
        json.dump(est, f, indent=2)


if __name__ == "__main__":
    main()
