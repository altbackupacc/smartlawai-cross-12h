"""Append-only annotation log, plus the blinding/randomisation/mode helpers the
app depends on, plus export straight to eval/gold/*.jsonl.

PLAN.md M6 requires the app to export "straight to eval/gold/*.jsonl -- no
manual spreadsheet-to-JSON step", which means the aggregation from N raters to
one gold row lives here rather than in a human's head.

Append-only on purpose: raw annotations are the evidence behind every agreement
statistic in the paper. Nothing in this module edits or deletes a submitted
annotation; a corrected judgment is a new row with a later timestamp, and
`latest_by_rater()` decides which one counts. That keeps the anchoring analysis
(RESEARCH.md §7.1.1) honest even if a rater changes their mind.

The pure helpers here (`blind_item`, `randomised_order`, `assign_mode`) are
separated from app.py deliberately: they carry the methodological guarantees
PLAN.md calls non-negotiable, and Streamlit callbacks are not testable.

Imports nothing from `smartlawai` (OPS.md §8).
"""
from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import Counter
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval import config
from eval.annotate.agreement import AgreementResult, agreement
from eval.annotate.cloud_storage import GCSAnnotationBackend, resolve_backend
from eval.schemas import GOLD_SET_SPECS, Annotation, SchemaError

# Fields that would reveal which system produced an item. Stripped before an
# item is ever rendered (PLAN.md M6: "a rater who knows which output is ours is
# not giving us data").
BLINDED_FIELDS: frozenset[str] = frozenset({
    "system", "system_name", "model", "model_id", "run_id", "arm", "condition",
    "baseline", "is_ours", "provenance", "proposed_by",
})


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------- #
# Methodological helpers -- pure, therefore testable
# --------------------------------------------------------------------------- #
def blind_item(item: dict[str, Any]) -> dict[str, Any]:
    """Strip every system-identifying field before display.

    Nested dicts are blinded too: a `detail` or `trace` blob is exactly where a
    model id hides.
    """
    out: dict[str, Any] = {}
    for key, value in item.items():
        if key in BLINDED_FIELDS:
            continue
        out[key] = blind_item(value) if isinstance(value, dict) else value
    return out


def randomised_order(item_ids: Iterable[str], rater_id: str,
                     seed: int = config.DEFAULT_SEED) -> list[str]:
    """Deterministic per-rater shuffle.

    Randomised so raters do not see items in corpus order (which correlates with
    document, and therefore with difficulty); deterministic per rater so a rater
    who closes the app and comes back resumes the same queue instead of
    re-rating items in a new order.
    """
    ids = list(item_ids)
    digest = hashlib.sha256(f"{seed}:{rater_id}".encode()).hexdigest()
    random.Random(int(digest[:16], 16)).shuffle(ids)
    return ids


def assign_mode(item_id: str, cold_fraction: float = config.COLD_SUBSET_FRACTION,
                seed: int = config.DEFAULT_SEED) -> str:
    """Deterministically assign an item to the cold or correction subset.

    RESEARCH.md §7.1.1's anchoring control needs a ~10% cold subset. Assignment
    is a hash of the item id, not a coin flip, so every rater sees the same item
    in the same mode and the cold subset is comparable across raters.
    """
    if not 0.0 <= cold_fraction <= 1.0:
        raise ValueError("cold_fraction must be in [0, 1]")
    digest = hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "cold" if bucket < cold_fraction else "correction"


# --------------------------------------------------------------------------- #
# The log
# --------------------------------------------------------------------------- #
@dataclass
class ThroughputStats:
    """RESEARCH.md §7.3: "Record the pilot's items/hour -- it is how you size
    everything else."""

    n_annotations: int
    median_seconds: float | None
    mean_seconds: float | None
    items_per_hour: float | None
    by_mode: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {"n_annotations": self.n_annotations,
                "median_seconds": self.median_seconds,
                "mean_seconds": self.mean_seconds,
                "items_per_hour": self.items_per_hour,
                "by_mode_items_per_hour": self.by_mode}


class AnnotationStore:
    """One JSONL file per task under eval/annotations/.

    Not a database: annotation volume is hundreds of rows, the file is the audit
    trail, and adding DuckDB here would put an eval concern in the serving
    backend's territory (`adapters/`, which M6 must not touch).

    The local file is always the read/write path used below -- that does not
    change. What can change is durability: when a GCS backend is configured
    (`SMARTLAW_ANNOTATIONS_BACKEND=gcs`; see `cloud_storage.py` and
    `docs/M6_PERSISTENT_STORAGE.md`), `append()` additionally uploads the row
    as its own immutable object and `load()` first pulls down any objects not
    yet mirrored locally -- so the local file becomes a synced cache of a
    durable remote log instead of the only copy, without changing a single
    line of the local-only read/write logic itself. With no cloud backend
    configured (the default), behaviour is identical to before this existed.
    """

    def __init__(self, task: str, root: Path | None = None,
                 cloud: GCSAnnotationBackend | None | str = "auto") -> None:
        if task not in GOLD_SET_SPECS:
            raise SchemaError(f"unknown task {task!r}; known: {sorted(GOLD_SET_SPECS)}")
        self.task = task
        self.root = root or config.ANNOTATIONS_DIR
        self.path = self.root / f"{task}.jsonl"
        self._cloud = resolve_backend() if cloud == "auto" else cloud

    # ---- write ----
    def append(self, annotation: Annotation) -> Annotation:
        annotation.validate()
        row = annotation.to_row()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if self._cloud is not None:
            self._cloud.upload_annotation(self.task, row)
        return annotation

    def record(self, item_id: str, rater_id: str, label: Any, mode: str,
               seconds_on_item: float, proposed_label: Any = None,
               comment: str = "", confidence: int | None = None,
               rubric_version: str = "", session_id: str = "") -> Annotation:
        """Build, validate and append one annotation in a single call."""
        accepted = None if mode == "cold" else (label == proposed_label)
        return self.append(Annotation(
            annotation_id=f"ann-{uuid.uuid4().hex[:12]}",
            task=self.task, item_id=item_id, rater_id=rater_id, label=label,
            mode=mode, created_at=_now_iso(), seconds_on_item=float(seconds_on_item),
            proposed_label=proposed_label, accepted_proposal=accepted,
            confidence=confidence, comment=comment,
            rubric_version=rubric_version, session_id=session_id))

    # ---- read ----
    def load(self) -> list[Annotation]:
        if self._cloud is not None:
            self._cloud.sync_to_local(self.task, self.path)
        if not self.path.exists():
            return []
        out: list[Annotation] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Annotation.from_row(json.loads(line)))
                except (json.JSONDecodeError, SchemaError) as e:
                    raise SchemaError(f"{self.path}:{lineno}: {e}") from e
        return out

    def by_item(self) -> dict[str, list[Annotation]]:
        grouped: dict[str, list[Annotation]] = {}
        for ann in self.load():
            grouped.setdefault(ann.item_id, []).append(ann)
        return grouped

    def latest_by_rater(self) -> dict[str, dict[str, Annotation]]:
        """{item_id: {rater_id: most recent annotation}} -- append-only means a
        rater can appear twice for one item; the later submission wins."""
        out: dict[str, dict[str, Annotation]] = {}
        for ann in self.load():
            current = out.setdefault(ann.item_id, {}).get(ann.rater_id)
            if current is None or ann.created_at >= current.created_at:
                out[ann.item_id][ann.rater_id] = ann
        return out

    def rated_item_ids(self, rater_id: str) -> set[str]:
        return {i for i, raters in self.latest_by_rater().items() if rater_id in raters}

    # ---- analysis ----
    def agreement(self) -> AgreementResult:
        """Live agreement over everything submitted so far (PLAN.md M6)."""
        labels: dict[Hashable, dict[Hashable, Hashable]] = {
            item_id: {rater: _hashable(ann.label) for rater, ann in raters.items()}
            for item_id, raters in self.latest_by_rater().items()}
        return agreement(labels)

    def throughput(self) -> ThroughputStats:
        annotations = self.load()
        times = [a.seconds_on_item for a in annotations if a.seconds_on_item > 0]
        if not times:
            return ThroughputStats(len(annotations), None, None, None, {})
        ordered = sorted(times)
        mid = len(ordered) // 2
        median = (ordered[mid] if len(ordered) % 2
                  else (ordered[mid - 1] + ordered[mid]) / 2)
        mean = sum(times) / len(times)
        by_mode: dict[str, float | None] = {}
        for mode in ("correction", "cold"):
            mode_times = [a.seconds_on_item for a in annotations
                          if a.mode == mode and a.seconds_on_item > 0]
            by_mode[mode] = (round(3600.0 / (sum(mode_times) / len(mode_times)), 2)
                             if mode_times else None)
        return ThroughputStats(
            n_annotations=len(annotations),
            median_seconds=round(median, 2),
            mean_seconds=round(mean, 2),
            items_per_hour=round(3600.0 / mean, 2),
            by_mode=by_mode)

    def anchoring_pairs(self) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
        """(assisted, cold) lists of (human_label, model_label).

        Cold-mode items carry no model proposal by construction, so the model's
        label for them has to come from the correction-mode proposal recorded
        for the SAME item by another rater. Items where no proposal exists
        anywhere are excluded -- there is nothing to compare against.
        """
        proposals: dict[str, Any] = {}
        for ann in self.load():
            if ann.proposed_label is not None:
                proposals.setdefault(ann.item_id, ann.proposed_label)

        assisted: list[tuple[Any, Any]] = []
        cold: list[tuple[Any, Any]] = []
        for ann in self.load():
            model_label = proposals.get(ann.item_id)
            if model_label is None:
                continue
            pair = (_hashable(ann.label), _hashable(model_label))
            (cold if ann.mode == "cold" else assisted).append(pair)
        return assisted, cold

    # ---- export ----
    def export_gold_set(self, items_by_id: dict[str, Any],
                        gold_dir: Path | None = None,
                        min_annotations: int = 1) -> list:
        """Fold annotations into gold items and write eval/gold/<task>.jsonl.

        `items_by_id` supplies the item bodies (query/passage/claim text); this
        method supplies the labels, rater ids, annotation count and agreement.

        Aggregation is majority vote with a deterministic tie-break (the label
        that sorts first), and the tie is recorded in `notes` rather than hidden
        -- a tied item is exactly the kind of ambiguity RESEARCH.md §7.3 says the
        pilot exists to surface.
        """
        from eval.gold_sets import write_gold_set  # local: avoids an import cycle

        latest = self.latest_by_rater()
        exported = []
        for item_id, raters in sorted(latest.items()):
            if len(raters) < min_annotations or item_id not in items_by_id:
                continue
            item = items_by_id[item_id]
            labels = [ann.label for ann in raters.values()]
            consensus, tied = _majority(labels)

            _apply_label(item, consensus)
            item.rater_ids = sorted(raters)
            item.n_annotations = len(raters)
            item.rubric_version = next(
                (a.rubric_version for a in raters.values() if a.rubric_version), "")
            if len(raters) >= 2:
                pair_labels = {item_id: {r: _hashable(a.label)
                                         for r, a in raters.items()}}
                item.agreement = agreement(pair_labels).value
            if tied:
                item.notes = (item.notes + " " if item.notes else "") + \
                    f"[tie among {sorted(set(map(str, labels)))}; resolved by sort order]"
            exported.append(item)

        write_gold_set(self.task, exported, gold_dir)
        return exported


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _hashable(label: Any) -> Hashable:
    """Graded retrieval labels are dicts; agreement needs something hashable."""
    if isinstance(label, dict):
        return json.dumps(label, sort_keys=True)
    if isinstance(label, list):
        return json.dumps(label)
    return label


def _majority(labels: list[Any]) -> tuple[Any, bool]:
    """Majority label plus whether it was a tie. Deterministic tie-break."""
    counts = Counter(_hashable(x) for x in labels)
    top = max(counts.values())
    winners = sorted((k for k, v in counts.items() if v == top), key=str)
    chosen = winners[0]
    original = next(x for x in labels if _hashable(x) == chosen)
    return original, len(winners) > 1


def _apply_label(item: Any, label: Any) -> None:
    """Write the consensus label onto whichever field this schema calls it."""
    if isinstance(label, dict) and hasattr(item, "judgments"):
        item.judgments = {k: int(v) for k, v in label.items()}
        return
    for attr in ("label", "in_force_label"):
        if hasattr(item, attr):
            setattr(item, attr, label)
            return
    raise SchemaError(f"cannot apply label to {type(item).__name__}: no label field")
