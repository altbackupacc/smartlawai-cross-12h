"""Orchestrator for M7's baseline conditions. Runs one condition over a list
of (question, scope) items, writes results to baselines/results/ (schema-
close to eval/run_eval.py's own output so eval.stats.paired_bootstrap can
compare a baseline's per-item scores against the real pipeline's without a
translation layer -- see M6_ONBOARDING.md 11 for that assumed contract).

Kept in a directory separate from eval/results/ so OPS.md's "every
eval/results/*.json is kept forever, in git, and every paper number traces
back to one" retention story isn't muddied by baseline runs.

Scoring via eval.metrics/eval.stats is imported lazily and guarded: if
eval/ isn't present in this checkout (true as of M7 being written -- M6 has
not landed here yet), `metrics` in the output is reported as
'unavailable: <reason>' rather than crashing the whole run (I2's
never-fail-open discipline applied to a missing dependency, not just a
missing score) -- the raw per-item generations and contamination split are
still produced and written, since neither needs eval/ to exist."""
from __future__ import annotations

import json
import time
from pathlib import Path

import click

from smartlawai.adapters.base import BackendInterface
from smartlawai.scope import Scope

from .contamination import resolve_judgment_date, split_by_cutoff
from .generators import FrontierGenerator, MistralGenerator, SaulLMGenerator
from .retrieval import scoped_bm25, scoped_hybrid

# condition -> (generator key, retrieval kind: "none" | "bm25" | "hybrid")
CONDITIONS: dict[str, tuple[str, str]] = {
    "mistral_zero_shot": ("mistral", "none"),
    "mistral_rag": ("mistral", "hybrid"),
    "saullm_zero_shot": ("saullm", "none"),
    "saullm_rag": ("saullm", "hybrid"),
    "bm25_frontier": ("frontier", "bm25"),
    "frontier_zero_shot": ("frontier", "none"),
    "frontier_rag": ("frontier", "hybrid"),
}

_GENERATORS = {"mistral": MistralGenerator, "saullm": SaulLMGenerator,
               "frontier": FrontierGenerator}


def _build_encoder_reranker(device: str | None):
    """Only constructed for hybrid-retrieval conditions -- real local GPU
    inference (InLegalBERT + cross-encoder), so this follows CLAUDE.md #2's
    'every training/inference script accepts --device'."""
    from smartlawai.core.inlegalbert import InLegalBERTEncoder
    from smartlawai.core.rerank import Reranker
    from train.common.device import resolve_device

    resolved = resolve_device(device)
    return InLegalBERTEncoder(device=str(resolved)), Reranker()


def _resolve_contamination(backend: BackendInterface, scope: Scope) -> str:
    """Best-effort: concatenates in-scope chunk text and resolves a date via
    the header heuristic (baselines/contamination.py). Uses the same scoped
    retrieval entry point as everything else here (I1)."""
    chunks = backend.fetch_chunks_scoped(scope)
    text = " ".join(c.chunk_text for c in chunks)
    doc_id = scope.doc_ids[0] if scope.doc_ids else "unknown"
    return resolve_judgment_date(doc_id, text)


def _try_score(items: list[dict]) -> dict | str:
    """Attempts real metric computation via eval.metrics/eval.stats. Returns
    an 'unavailable: <reason>' string rather than raising if that package
    isn't present -- expected in this checkout since M6 hasn't landed here."""
    try:
        from eval.metrics import rouge_scores  # noqa: F401
        from eval.stats import paired_bootstrap  # noqa: F401
    except ImportError as e:
        return f"unavailable: eval package not present in this checkout ({e})"
    # Real scoring needs gold references from eval/gold/*.jsonl, which this
    # function's caller doesn't have access to yet either -- left for the
    # eval-harness integration once M6's gold-set loader exists.
    return "unavailable: gold references not wired in yet (pending M6 gold-set loader)"


def run(condition: str, questions: list[tuple[str, Scope]],
        backend: BackendInterface | None = None, device: str | None = None,
        model_id_for_cutoff: str | None = None,
        out_dir: Path = Path("baselines/results")) -> dict:
    """One baseline condition over a list of (question, scope) items -- the
    shape a real gold-set loader (eval.gold_sets, per the assumed M6 contract)
    would hand this. Mirrors eval/run_eval.py's run() signature intentionally
    so the two orchestrators stay structurally comparable."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}. Options: {sorted(CONDITIONS)}")
    model_key, retrieval_kind = CONDITIONS[condition]
    generator = _GENERATORS[model_key]()

    encoder = reranker = None
    if retrieval_kind == "hybrid":
        encoder, reranker = _build_encoder_reranker(device)

    if backend is None:
        from smartlawai.adapters.factory import get_backend
        backend = get_backend()

    per_item = []
    date_resolutions = []
    for question, scope in questions:
        if retrieval_kind == "none":
            passages = []
        elif retrieval_kind == "bm25":
            passages = scoped_bm25(backend, scope, question)
        else:
            passages = scoped_hybrid(backend, scope, question, encoder, reranker)

        result = generator.generate(question, passages)
        per_item.append({
            "question": question, "model_id": result.model_id,
            "claims": [{"text": c.text, "passage_ids": c.passage_ids,
                        "citations": c.citations} for c in result.claims],
            "unanswerable_aspects": result.unanswerable_aspects,
        })
        date_resolutions.append(_resolve_contamination(backend, scope))

    contamination = None
    if model_id_for_cutoff is not None:
        split = split_by_cutoff(date_resolutions, model_id_for_cutoff)
        contamination = {k: [r.doc_id for r in v] for k, v in split.items()}

    out = {
        "condition": condition,
        "n_items": len(per_item),
        "items": per_item,
        "contamination": contamination,
        "metrics": _try_score(per_item),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{condition}_{int(time.time())}.json"
    out_path.write_text(json.dumps(out, indent=2))
    return out


@click.command()
@click.option("--condition", required=True, type=click.Choice(sorted(CONDITIONS)))
@click.option("--device", default=None, help="cpu | cuda | auto (default: auto-detect)")
def main(condition: str, device: str | None) -> None:
    """CLI entry point. Loading real (question, scope) items from
    eval/gold/*.jsonl is left to the (assumed) M6 gold-set loader -- wire that
    in once M6 lands in this checkout; this raises clearly rather than
    silently running on no data."""
    raise NotImplementedError(
        "Wire this up to eval's gold-set loader once M6 lands in this "
        "checkout, then call run(condition, questions, device=device).")


if __name__ == "__main__":
    main()
