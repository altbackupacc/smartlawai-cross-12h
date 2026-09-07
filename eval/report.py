"""Turn a run's results dict into the two artefacts M6 owes its readers.

Reference image 2 lists "evaluation report (tables + results)" as a first-class
M6 deliverable, so a results file alone is not the deliverable:

  * a machine-readable JSON, kept forever in git (OPS.md §4's retention table),
    which is what every number in the eventual paper traces back to; and
  * a Markdown report a human can read without opening the JSON.

The single rule that shapes this module: **a metric that was not measured must
look different from a metric that measured zero.** An `unavailable` metric
renders as `unavailable` with its reason, never as `0.0` or a blank cell, and
the report carries a dedicated section listing everything that could not be
computed and why. That section is expected to be long right now -- most of the
pipeline is stubbed -- and a long honest one is the point (M6_ONBOARDING.md
§11: "It doesn't need to look good. It needs to run and be honest.").

Imports nothing from `smartlawai` (OPS.md §8).
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from eval.metrics import MetricResult

UNAVAILABLE = "_unavailable_"


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return UNAVAILABLE
    return f"{value:.{digits}f}"


def _metric_rows(metrics: Iterable[dict]) -> list[str]:
    rows = ["| Metric | Value | n | Status | Reason |",
            "|---|---:|---:|---|---|"]
    for m in metrics:
        rows.append(f"| `{m['name']}` | {_fmt(m.get('value'))} | {m.get('n', 0)} | "
                    f"{m.get('status', 'ok')} | {m.get('reason', '') or '—'} |")
    return rows


def as_dicts(metrics: Iterable[Any]) -> list[dict]:
    """Accept MetricResult objects or already-serialised dicts."""
    return [m.to_dict() if isinstance(m, MetricResult) else dict(m) for m in metrics]


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def render_markdown(results: dict) -> str:
    """Render the human-readable report."""
    manifest = results.get("manifest", {})
    sections = results.get("sections", {})
    lines: list[str] = []

    lines.append(f"# SmartLawAI evaluation report — `{manifest.get('split', '?')}`")
    lines.append("")
    lines.append(f"Generated {manifest.get('generated_at', '?')} · "
                 f"seed {manifest.get('seed', '?')} · "
                 f"harness {manifest.get('harness_version', '?')}")
    lines.append("")

    # ---- system configuration ----
    lines.append("## System configuration")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    for key in ("split", "seed", "backend", "generator_model", "encoder", "reranker",
                "verifier", "pipeline_is_stubbed", "git_commit", "python",
                "n_items_run"):
        if key in manifest:
            lines.append(f"| {key} | `{manifest[key]}` |")
    lines.append("")
    if manifest.get("pipeline_is_stubbed"):
        lines.append("> **Every model in this run is a stub (M0).** These numbers "
                     "measure that the harness computes end to end, not system "
                     "quality. That is the M6 done-when criterion.")
        lines.append("")

    # ---- gold sets ----
    gold = results.get("gold_sets", {})
    if gold:
        lines.append("## Evaluation datasets")
        lines.append("")
        lines.append("| Set | n | Target n | 95% CI half-width @ p=0.5 | "
                     "IAA duplicated | target | mean agreement | sha256 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for name, stats in sorted(gold.items()):
            lines.append(
                f"| `{name}` | {stats['n']} | {stats['target_n']} | "
                f"{_fmt(stats.get('ci_halfwidth_at_p50'))} | "
                f"{stats.get('duplication_rate_achieved', 0):.0%} | "
                f"{stats.get('duplication_rate_target', 0):.0%} | "
                f"{_fmt(stats.get('mean_agreement'))} | "
                f"`{(stats.get('sha256') or '')[:12]}` |")
        lines.append("")
        for note in results.get("sizing_notes", []):
            lines.append(f"- {note}")
        lines.append("")

    missing = results.get("missing_gold_sets", {})
    if missing:
        lines.append("### Gold sets not available")
        lines.append("")
        for name, reason in sorted(missing.items()):
            lines.append(f"- **`{name}`** — {reason}")
        lines.append("")

    # ---- metric sections ----
    titles = {
        "retrieval": "Retrieval metrics",
        "answer": "Answer quality and groundedness",
        "citation": "Citation and authority metrics",
        "robustness": "Robustness — out-of-scope and refusal",
        "calibration": "Calibration",
        "cost": "Cost and latency",
        "human": "Human evaluation",
    }
    for key, title in titles.items():
        metrics = sections.get(key)
        if not metrics:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(_metric_rows(metrics))
        lines.append("")

    # ---- calibration detail ----
    bins = results.get("reliability_diagram")
    if bins:
        lines.append("### Reliability diagram data")
        lines.append("")
        lines.append("| Bin | Mean confidence | Accuracy | Count |")
        lines.append("|---|---:|---:|---:|")
        for b in bins:
            lines.append(f"| [{b['bin_lower']:.1f}, {b['bin_upper']:.1f}) | "
                         f"{b['mean_confidence']:.4f} | {b['accuracy']:.4f} | "
                         f"{b['count']} |")
        lines.append("")

    # ---- failures ----
    failures = results.get("failures", [])
    lines.append("## Failures")
    lines.append("")
    if not failures:
        lines.append("No item raised an error during this run.")
    else:
        lines.append(f"{len(failures)} item(s) failed:")
        lines.append("")
        lines.append("| Item | Stage | Error |")
        lines.append("|---|---|---|")
        for f in failures[:50]:
            lines.append(f"| `{f.get('item_id', '?')}` | {f.get('stage', '?')} | "
                         f"{str(f.get('error', ''))[:160]} |")
        if len(failures) > 50:
            lines.append("")
            lines.append(f"_...and {len(failures) - 50} more; see the JSON._")
    lines.append("")

    # ---- error analysis ----
    analysis = results.get("error_analysis", {})
    if analysis:
        lines.append("## Error analysis")
        lines.append("")
        for group, counts in sorted(analysis.items()):
            lines.append(f"**{group.replace('_', ' ')}**")
            lines.append("")
            lines.append("| Reason | Count |")
            lines.append("|---|---:|")
            for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"| `{reason}` | {count} |")
            lines.append("")

    # ---- what could not be measured ----
    unavailable = [m for metrics in sections.values() for m in metrics
                   if m.get("status") != "ok"]
    lines.append("## Not measured")
    lines.append("")
    if not unavailable:
        lines.append("Every applicable metric computed.")
    else:
        lines.append("These metrics did not produce a number. They are reported as "
                     "`unavailable`, never as `0.0` — CLAUDE.md I2.")
        lines.append("")
        lines.append("| Metric | Status | Reason |")
        lines.append("|---|---|---|")
        for m in unavailable:
            lines.append(f"| `{m['name']}` | {m['status']} | "
                         f"{m.get('reason', '') or '—'} |")
    lines.append("")

    # ---- aggregate ----
    summary = results.get("summary", {})
    if summary:
        lines.append("## Aggregate summary")
        lines.append("")
        lines.append("| | |")
        lines.append("|---|---:|")
        for key, value in sorted(summary.items()):
            lines.append(f"| {key.replace('_', ' ')} | {value} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Produced by `eval/run_eval.py`. Every number here is computed, "
                 "never hand-typed; this file and its `.json` sibling are retained "
                 "in git permanently (OPS.md §4).")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_report(results: dict, out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write `<stem>.json` and `<stem>.md`. Returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False,
                                    sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(results), encoding="utf-8")
    return json_path, md_path


def flat_metrics(results: dict) -> dict[str, float]:
    """{metric_name: value} for every metric that actually produced a number.

    General-purpose: used for the CI regression gate (via
    `compare_to_baseline`, which filters this further) and anywhere else a
    flat view of "everything measured" is useful. Unavailable metrics are
    simply absent, so a metric going from `unavailable` to a number (or back)
    shows up as an appearance/disappearance rather than a silent comparison
    against a fabricated zero.
    """
    out: dict[str, float] = {}
    for metrics in results.get("sections", {}).values():
        for m in metrics:
            if m.get("status") == "ok" and m.get("value") is not None:
                out[m["name"]] = float(m["value"])
    return out


def _is_quality_metric(name: str) -> bool:
    """Excludes wall-clock timing from the regression gate.

    Found during the M6 audit (docs/M6_GAP_ANALYSIS.md): `latency_p50_ms[...]`
    /`latency_p95_ms[...]` are genuine measurements, but they are real
    wall-clock numbers that legitimately jitter between runs on the same
    machine (confirmed directly: two back-to-back runs against the same gold
    data produced retrieve-stage p50s of 0.78ms and 0.62ms). Comparing them
    against a single committed baseline with a uniform absolute tolerance
    produces a regression-gate false positive on almost every run, which
    trains reviewers to ignore the gate -- defeating M6_ONBOARDING.md §11's
    purpose for it ("guarding against the harness silently breaking, not
    against real quality regressions"). Latency is still reported in every
    results file; it is only excluded from the pass/fail regression check.
    """
    return not name.startswith("latency_")


def compare_to_baseline(current: dict, baseline: dict,
                        tolerance: float) -> tuple[list[dict], list[dict]]:
    """Compare two results files. Returns (regressions, disappeared).

    A regression is a metric that dropped by more than `tolerance` in absolute
    terms. `disappeared` lists metrics the baseline measured and this run did
    not -- a harness that stops computing a metric is exactly the silent
    breakage this gate exists to catch, and it would otherwise pass unnoticed.
    Both checks exclude timing metrics (`_is_quality_metric`) -- see its
    docstring.
    """
    now = {k: v for k, v in flat_metrics(current).items() if _is_quality_metric(k)}
    before = {k: v for k, v in flat_metrics(baseline).items() if _is_quality_metric(k)}
    regressions = [
        {"metric": name, "baseline": before[name], "current": now[name],
         "delta": round(now[name] - before[name], 6)}
        for name in sorted(before) if name in now
        and now[name] < before[name] - tolerance
    ]
    disappeared = [{"metric": name, "baseline": before[name]}
                   for name in sorted(before) if name not in now]
    return regressions, disappeared
