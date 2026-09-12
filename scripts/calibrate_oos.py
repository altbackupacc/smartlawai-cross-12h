"""Fit Platt scaling calibration curve on eval/gold/oos.jsonl (PLAN.md M5.6).

Scores each OOS item by running the *real* retrieval + rerank path
(InLegalBERTEncoder + Reranker, scoped per I1) against the judgment its
`source_doc_id`/`doc_ids` point to, and fitting core.oos.fit_platt_scaling on
the resulting (top reranker logit, in_scope label) pairs.

Per CLAUDE.md #6, this script downloads model weights and is a
measured/reported result -- it is drafted here but run by a human (on GCP L4
per CLAUDE.md #2's routing table, or locally if the InLegalBERT/reranker
weights are already cached), not executed by the agent loop.

I2 discipline: every eval/gold/oos.jsonl seed row currently carries empty
doc_ids/source_doc_id (see eval/gold/PROVENANCE.md -- the same "no stable
corpus with stable ids yet" gap documented for retrieval.jsonl). An item with
no resolvable document has nothing to retrieve against, so it is skipped and
counted, never scored with a synthetic stand-in. If zero items resolve, this
script refuses to write a scaler and reports why, instead of writing a
plausible-looking PlattScaler fit on invented numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.gold_sets import load_gold_set
from smartlawai.adapters.factory import get_backend
from smartlawai.core.oos import fit_platt_scaling
from smartlawai.scope import Scope

OUT_FILE = REPO_ROOT / "data" / "processed" / "oos_calibration.json"


def _top_rerank_logit(be, encoder, reranker, question: str, doc_id: str) -> float | None:
    """Real retrieval + rerank against one document; returns the top cross-encoder
    logit, or None if the document has no indexed chunks to score against."""
    scope = Scope(doc_ids=(doc_id,), owner_id="oos-calibration")
    candidates = be.fetch_chunks_scoped(scope)
    if not candidates:
        return None

    encoder.embed([question])  # proves the retrieval seam is live before reranking
    ranked = reranker.rerank(question, candidates, top_k=1)
    if not ranked:
        return None
    return float(ranked[0].score)


def main() -> int:
    oos_items = load_gold_set("oos")
    print(f"Loaded {len(oos_items)} OOS evaluation items.")

    be = get_backend()

    from smartlawai.core.inlegalbert import InLegalBERTEncoder
    from smartlawai.core.rerank import Reranker

    encoder = InLegalBERTEncoder()
    reranker = Reranker()

    logits: list[float] = []
    labels: list[int] = []
    skipped_no_doc_id = 0
    skipped_no_chunks = 0

    for item in oos_items:
        doc_id = (item.doc_ids[0] if getattr(item, "doc_ids", None) else "") or getattr(
            item, "source_doc_id", ""
        )
        if not doc_id:
            skipped_no_doc_id += 1
            continue

        logit = _top_rerank_logit(be, encoder, reranker, item.question, doc_id)
        if logit is None:
            skipped_no_chunks += 1
            continue

        logits.append(logit)
        labels.append(1 if item.label == "in_scope" else 0)

    n_scored = len(logits)
    print(f"Scored {n_scored} items via real retrieval+rerank; "
          f"skipped {skipped_no_doc_id} with no doc_id, "
          f"{skipped_no_chunks} whose document has no indexed chunks.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if n_scored < 10:
        # I2: refuse to fit and publish a scaler on too little (or zero) real
        # signal rather than writing something that looks calibrated.
        report = {
            "calibration_status": "unavailable",
            "reason": (
                "fewer than 10 oos.jsonl items resolved to an indexed document "
                "(see eval/gold/PROVENANCE.md's 'no stable corpus with stable ids "
                "yet' note) -- calibration needs items with a real doc_id/chunks "
                "to retrieve against, not a synthetic stand-in"
            ),
            "n_items": len(oos_items),
            "n_scored": n_scored,
            "skipped_no_doc_id": skipped_no_doc_id,
            "skipped_no_chunks": skipped_no_chunks,
        }
        OUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote unavailable status to {OUT_FILE} -- see 'reason'.")
        return 1

    scaler = fit_platt_scaling(logits, labels, learning_rate=0.1, max_iter=200)

    in_scope_probs = [scaler.predict_proba(logit) for logit, lab in zip(logits, labels) if lab == 1]
    oos_probs = [scaler.predict_proba(logit) for logit, lab in zip(logits, labels) if lab == 0]
    mean_in_scope = sum(in_scope_probs) / len(in_scope_probs) if in_scope_probs else 0.0
    mean_oos = sum(oos_probs) / len(oos_probs) if oos_probs else 0.0

    report = {
        "calibration_status": "fitted",
        "scaler": scaler.to_dict(),
        "n_items": len(oos_items),
        "n_scored": n_scored,
        "skipped_no_doc_id": skipped_no_doc_id,
        "skipped_no_chunks": skipped_no_chunks,
        "mean_in_scope_prob": round(mean_in_scope, 4),
        "mean_oos_prob": round(mean_oos, 4),
    }
    OUT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote calibrated OOS parameters to {OUT_FILE}")
    print(f"Mean In-Scope Probability: {mean_in_scope:.4f}")
    print(f"Mean OOS Probability: {mean_oos:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
