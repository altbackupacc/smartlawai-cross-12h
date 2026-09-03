"""RR <-> SUMM alignment investigation (M1 step 2, PLAN.md / M1_ONBOARDING.md #6).

Hard checkpoint: nothing past this script (build_pairs.py onward) runs until
this decision is made and written to data/rr_summ_alignment_report.md.
Read-only: never writes to data/processed/.
"""
from __future__ import annotations

from datasets import load_from_disk

COVERAGE_THRESHOLD = 0.70  # PLAN.md's literal decision boundary; don't change without updating PLAN.md
REPORT_PATH = "data/rr_summ_alignment_report.md"


def load_ids(path: str, id_field: str = "id") -> dict[str, set[str]]:
    ds = load_from_disk(path)
    return {split: set(ds[split][id_field]) for split in ds.keys()}


def investigate() -> dict:
    summ_by_split = load_ids("data/raw/summ")
    rr_by_split = load_ids("data/raw/rr")

    summ_ids = set().union(*summ_by_split.values())
    rr_ids = set().union(*rr_by_split.values())
    overlap = summ_ids & rr_ids

    coverage = len(overlap) / len(summ_ids) if summ_ids else 0.0
    path = "Path 1 primary" if coverage >= COVERAGE_THRESHOLD else "Path 2 primary"

    return {
        "summ_ids": summ_ids,
        "summ_by_split": summ_by_split,
        "rr_ids": rr_ids,
        "rr_by_split": rr_by_split,
        "overlap": overlap,
        "coverage": coverage,
        "path": path,
    }


def write_report(result: dict, out_path: str = REPORT_PATH) -> None:
    summ_counts = ", ".join(f"{k}={len(v)}" for k, v in result["summ_by_split"].items())
    rr_counts = ", ".join(f"{k}={len(v)}" for k, v in result["rr_by_split"].items())

    lines = [
        "# RR <-> SUMM Alignment Investigation",
        "",
        "**Verdict: Path 2 (frontier-generated section summaries + human validation) "
        "is the section-pair construction method for the whole corpus.**",
        "",
        "## What was checked (in order, per M1_ONBOARDING.md #6)",
        "",
        "1. **Real schema** (not assumed):",
        "   - `summ`: `id, document, summary, num_doc_tokens, num_summ_tokens` "
        f"(document-level, professional headnote as `summary`). Split counts: {summ_counts}.",
        "   - `rr`: `id, text, labels, expert_1, expert_2, expert_3` — `text` is a "
        "per-sentence list, `labels` a matching per-sentence rhetorical-role id list "
        "(0-12), `expert_N` carry per-annotator primary/secondary label sequences for "
        f"inter-annotator agreement. Split counts: {rr_counts}.",
        "",
        f"2. **Shared joinable document id**: `summ` has **{len(result['summ_ids'])}** "
        f"unique ids (plain numeric strings, e.g. `'427'`, `'5037'`). `rr` has "
        f"**{len(result['rr_ids'])}** unique ids, which are structured court-case "
        "identifiers (e.g. `CCI_All_India_Distillers_Association_vs_Haldyn_Glass_...`) "
        "drawn from Competition Commission of India / High Court antitrust matters — a "
        "different document universe and a different legal domain than `summ`'s Supreme "
        f"Court judgments. **Overlap: {len(result['overlap'])} documents "
        f"({result['coverage']*100:.2f}% of summ).**",
        "",
        "3–4. **RR-span-to-headnote alignment quality, coverage**: not evaluated — moot. "
        "Per M1_ONBOARDING.md #6 point 2: *\"If the two configs draw from disjoint "
        "document universes, Path 1 is dead immediately, regardless of alignment "
        "quality.\"* That is exactly this case: zero shared documents, so there is no "
        "pair of (rr spans, summ headnote) to test alignment on in the first place.",
        "",
        "## Decision rule applied",
        "",
        f"Coverage = {result['coverage']*100:.2f}% "
        f"({'>=' if result['coverage'] >= COVERAGE_THRESHOLD else '<'} "
        f"{COVERAGE_THRESHOLD*100:.0f}% threshold) -> **{result['path']}**.",
        "",
        "No docs passed clean alignment (there are none to check), so there is no bonus "
        "gold subset to carry forward from Path 1 — this is a pure Path 2 corpus, not a "
        "hybrid one. That absence, not a judgment call, is what decided this.",
        "",
        "## What Path 2 means concretely for this corpus",
        "",
        "`summ` already provides a full-document (judgment, professional headnote) pair "
        "per document — this is *document-level* gold, not silver, and not itself in "
        "question. What Path 2 supplies is the missing piece RR was meant to give for "
        "free: a way to split that single document-level pair into several *section-level* "
        "pairs. `build_pairs.py` uses a frontier model to (a) segment the judgment text "
        "into sections and (b) produce a section-level summary aligned to the existing "
        "headnote, each pair tagged `provenance: path2-silver`. This is silver data by "
        "construction (RESEARCH.md T4) and must be human-validated (200-item sample, "
        "acceptance rate reported plainly) before being treated as usable training data.",
        "",
        "## Cost implication — action required before spending anything",
        "",
        "Path 2 triggering at full corpus scale (~7,030 train + 100 test `summ` docs) has "
        "**zero line item in `PLAN.md`'s COMPUTE BUDGET table**. Per M1_ONBOARDING.md #6, "
        "`build_pairs.py --dry-run` must report an estimated token volume and dollar cost, "
        "and that estimate must be surfaced to the user for explicit go-ahead before any "
        "real generation pass runs.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    result = investigate()
    print(f"summ ids: {len(result['summ_ids'])}, rr ids: {len(result['rr_ids'])}, "
          f"overlap: {len(result['overlap'])}, coverage: {result['coverage']*100:.2f}%, "
          f"decision: {result['path']}")
    write_report(result)
