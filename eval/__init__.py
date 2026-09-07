"""M6 — the evaluation harness and the gold sets it scores against.

Layering (OPS.md §8: "the eval harness is pure functions over data structures.
Neither imports the pipeline"):

    schemas / metrics / stats / extractiveness / annotate.agreement
        -> import NOTHING from smartlawai. Pure functions over dicts and lists.
    gold_sets / annotate.store / report
        -> file IO over those data structures. Still no smartlawai import.
    run_eval
        -> the ONE module allowed to import smartlawai.Pipeline, because
           M6_ONBOARDING.md §11 requires it to call Pipeline.ask() unmodified.

tests/test_eval_integration.py::test_metric_layer_does_not_import_smartlawai
enforces the top two layers. Breaking that boundary is a bug, not a refactor.
"""
from __future__ import annotations

__all__ = ["config", "extractiveness", "gold_sets", "metrics", "schemas", "stats"]
