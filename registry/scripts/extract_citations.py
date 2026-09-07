"""Demand-driven citation frequency analysis over the judgment corpus.

Reads IL-TUR SUMM (`data/raw/summ`, ~7.1k Supreme Court judgments), runs
`smartlawai.core.ner.extract_entities()` over each judgment, and counts which
statutes and sections are actually cited. That count is what the coverage
percentage is computed against (PLAN.md M4, threat T6).

Two correctness details that would otherwise quietly skew the numbers:

1. **Overlapping spans.** `ner.py` emits BOTH a STATUTE entity ("Section 420
   IPC") and a PROVISION entity ("Section 420") for the same text. Counting
   both double-counts every attached citation. A PROVISION whose span sits
   inside a STATUTE span is dropped as already-attributed.

2. **Unattached provisions.** A bare "Section 420" appearing after the act was
   last named comes back as PROVISION with no statute (M4_ONBOARDING.md 4).
   Undercounting these deflates per-statute frequency, so each is attributed to
   the nearest preceding STATUTE mention *in the same paragraph*. The
   attribution method is recorded per row so its effect can be measured rather
   than assumed.

Usage:
    python M4/scripts/extract_citations.py [--limit N] [--procs N]
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from registry.normalise import (
    is_compatible,
    normalise_act,
    parse_section_ref,
    section_id,
)

CORPUS = REPO / "data" / "raw" / "summ"
OUT = REPO / "registry" / "out"

# Gap (characters) within which attributing a bare provision to the preceding
# act mention is treated as high-confidence. Beyond it the row is kept but
# flagged, so the coverage number can be reported both ways.
NEAR_WINDOW = 600

_ds = None


def _init_worker(corpus_path: str) -> None:
    global _ds
    from datasets import load_from_disk

    _ds = load_from_disk(corpus_path)


def _doc_text(rec) -> str:
    """SUMM stores `document` as a list of paragraph strings."""
    doc = rec.get("document")
    if isinstance(doc, list):
        return "\n\n".join(x for x in doc if isinstance(x, str))
    return doc or ""


def _process(args):
    """Extract citation rows from one judgment. Returns (rows, stats)."""
    split, idx = args
    from smartlawai.core.ner import extract_entities

    rec = _ds[split][idx]
    doc_id = f"{split}:{rec.get('id', idx)}"
    text = _doc_text(rec)
    if not text:
        return [], Counter({"empty_docs": 1})

    # Paragraph start offsets, for "same paragraph" containment.
    para_starts = [0]
    for i in range(len(text) - 1):
        if text[i] == "\n" and text[i + 1] == "\n":
            para_starts.append(i + 2)

    def para_of(pos: int) -> int:
        return bisect.bisect_right(para_starts, pos) - 1

    res = extract_entities(doc_id, text)
    statutes = sorted(
        (e for e in res.entities if e.label == "STATUTE" and e.start >= 0),
        key=lambda e: e.start,
    )
    provisions = [e for e in res.entities if e.label == "PROVISION" and e.start >= 0]

    stats = Counter({"docs": 1})
    rows = []

    # --- STATUTE entities: may or may not carry a section ---
    for e in statutes:
        act = normalise_act(e.text)
        ref = parse_section_ref(e.text)
        if act and ref:
            rows.append(
                (doc_id, act, section_id(act, ref), ref.kind, ref.number,
                 "statute_span", e.text, 0, True)
            )
            stats["attached"] += 1
        elif act:
            rows.append((doc_id, act, None, None, None, "act_only", e.text, None, False))
            stats["act_only"] += 1
        else:
            stats["unrecognised_act"] += 1

    # --- PROVISION entities: drop those already inside a STATUTE span ---
    s_starts = [e.start for e in statutes]
    for p in provisions:
        j = bisect.bisect_right(s_starts, p.start) - 1
        if j >= 0 and statutes[j].start <= p.start and p.end <= statutes[j].end:
            stats["provision_inside_statute"] += 1
            continue

        ref = parse_section_ref(p.text)
        if ref is None:
            stats["provision_unparsed"] += 1
            continue

        # Nearest preceding STATUTE mention anywhere earlier in the document,
        # with the gap recorded. Measured on this corpus, the median gap is
        # ~4.7k chars: SUMM's `document` is a list of SENTENCES, and judgments
        # name the act once then cite bare sections for pages afterwards. A
        # same-paragraph window (what M4_ONBOARDING.md 4 suggests) attributes
        # almost nothing here. Distance is kept per row so confidence can decay
        # with it and the resolver can refuse beyond a threshold -- attributing
        # across 5k chars is a guess, and CLAUDE.md I2 forbids dressing a guess
        # as an answer.
        # Skip candidate acts that structurally cannot contain this kind of
        # provision, rather than taking the first act mention we find.
        act, dist = None, None
        k = bisect.bisect_left(s_starts, p.start) - 1
        while k >= 0:
            cand = statutes[k]
            a = normalise_act(cand.text)
            if a and is_compatible(a, ref.kind):
                act, dist = a, p.start - cand.end
                break
            if a:
                stats["fallback_kind_mismatch"] += 1
            k -= 1

        if act is not None:
            rows.append(
                (doc_id, act, section_id(act, ref), ref.kind, ref.number,
                 "nearest_preceding", p.text, dist, para_of(p.start) == para_of(p.start - dist))
            )
            stats["fallback_attributed"] += 1
            if dist <= NEAR_WINDOW:
                stats["fallback_near"] += 1
            else:
                stats["fallback_far"] += 1
        elif ref.kind == "art":
            # In Indian judgments a bare "Article N" is the Constitution in
            # practice. Recorded as its own attribution class so the assumption
            # is auditable rather than buried in the totals.
            rows.append(
                (doc_id, "constitution-1950", section_id("constitution-1950", ref),
                 ref.kind, ref.number, "article_default", p.text, None, False)
            )
            stats["article_default"] += 1
        else:
            rows.append(
                (doc_id, None, None, ref.kind, ref.number, "orphan", p.text, None, False)
            )
            stats["orphan"] += 1

    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap docs per split (smoke test)")
    ap.add_argument("--procs", type=int, default=max((os.cpu_count() or 4) - 2, 1))
    args = ap.parse_args()

    if not CORPUS.exists():
        print(f"ERROR: corpus not found at {CORPUS}", file=sys.stderr)
        return 1

    from datasets import load_from_disk

    ds = load_from_disk(str(CORPUS))
    tasks = []
    for split in ds:
        n = len(ds[split])
        if args.limit:
            n = min(n, args.limit)
        tasks.extend((split, i) for i in range(n))

    print(f"corpus  : {CORPUS}")
    print("splits  : %s" % {s: len(ds[s]) for s in ds})
    print("tasks   : %d docs   procs: %d" % (len(tasks), args.procs))

    import multiprocessing as mp

    all_rows, stats = [], Counter()
    with mp.Pool(args.procs, initializer=_init_worker, initargs=(str(CORPUS),)) as pool:
        for i, (rows, st) in enumerate(pool.imap_unordered(_process, tasks, chunksize=16), 1):
            all_rows.extend(rows)
            stats.update(st)
            if i % 500 == 0:
                print("  %d/%d docs, %d citation rows" % (i, len(tasks), len(all_rows)))

    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame(
        all_rows,
        columns=["doc_id", "act_id", "section_id", "kind", "number", "attribution",
                 "raw", "distance", "same_para"],
    )
    df.to_parquet(OUT / "citations.parquet", index=False)

    # --- frequency tables ---
    # "strict" = direct spans and near-window attributions only; the honest
    # lower bound. "wide" = everything we attributed at all.
    strict_mask = (df.attribution == "statute_span") | (
        (df.attribution == "nearest_preceding") & (df.distance <= NEAR_WINDOW)
    )
    sec = Counter(df.loc[df.section_id.notna(), "section_id"])
    sec_strict = Counter(df.loc[strict_mask & df.section_id.notna(), "section_id"])
    acts = Counter(df.loc[df.act_id.notna(), "act_id"])
    total_sec = sum(sec.values())

    cum, n95 = 0, 0
    for _, c in sec.most_common():
        cum += c
        n95 += 1
        if total_sec and cum / total_sec >= 0.95:
            break

    summary = {
        "docs_processed": stats["docs"],
        "citation_rows": len(df),
        "section_level_citations": total_sec,
        "distinct_sections": len(sec),
        "distinct_acts": len(acts),
        "sections_for_95pct_mass": n95,
        "section_citations_strict": sum(sec_strict.values()),
        "distinct_sections_strict": len(sec_strict),
        "near_window_chars": NEAR_WINDOW,
        "attribution_breakdown": {
            k: stats[k]
            for k in [
                "attached",
                "fallback_attributed",
                "fallback_near",
                "fallback_far",
                "fallback_kind_mismatch",
                "article_default",
                "act_only",
                "orphan",
                "provision_inside_statute",
                "provision_unparsed",
                "unrecognised_act",
            ]
        },
        "top_acts": acts.most_common(20),
        "top_sections": sec.most_common(40),
        "top_sections_strict": sec_strict.most_common(40),
    }
    (OUT / "citation_frequency.json").write_text(json.dumps(summary, indent=2))

    print("\n==== SUMMARY ====")
    print("docs                     : %d" % stats["docs"])
    print("citation rows            : %d" % len(df))
    print("section-level citations  : %d" % total_sec)
    print("distinct sections        : %d" % len(sec))
    print("sections for 95%% mass    : %d" % n95)
    print("  of which strict       : %d (%d distinct)" % (sum(sec_strict.values()), len(sec_strict)))
    print("attached (direct)        : %d" % stats["attached"])
    print("fallback attributed      : %d  (near<=%d: %d, far: %d)"
          % (stats["fallback_attributed"], NEAR_WINDOW, stats["fallback_near"], stats["fallback_far"]))
    print("article -> Constitution  : %d" % stats["article_default"])
    print("orphan (no act)          : %d" % stats["orphan"])
    print("act-only mentions        : %d" % stats["act_only"])
    print("\ntop acts:")
    for a, c in acts.most_common(12):
        print("  %-22s %d" % (a, c))
    print("\ntop sections:")
    for s, c in sec.most_common(15):
        print("  %-28s %d" % (s, c))
    print("\nwrote {} and {}".format(OUT / "citations.parquet", OUT / "citation_frequency.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
