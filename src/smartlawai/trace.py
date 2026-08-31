"""PipelineTrace: the observability + reproducibility record. Every pipeline
stage appends to it. No silent steps -- a stage missing from the trace is
missing forever (PLAN.md M0)."""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from smartlawai.scope import Scope


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class StageRecord:
    name: str
    status: str = "ok"  # "ok" | "error" | "unavailable" | "skipped"
    started_at: str = field(default_factory=_now_iso)
    duration_ms: float = 0.0
    detail: dict = field(default_factory=dict)  # JSON-primitive values only
    error: str | None = None


@dataclass
class PipelineTrace:
    trace_id: str
    scope: Scope | None
    stages: list[StageRecord] = field(default_factory=list)
    retrieval: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)
    verification: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)

    @contextmanager
    def stage(self, name: str) -> Iterator[StageRecord]:
        t0 = time.perf_counter()
        rec = StageRecord(name=name)
        try:
            yield rec
        except Exception as e:
            rec.status, rec.error = "error", str(e)
            raise
        finally:
            rec.duration_ms = (time.perf_counter() - t0) * 1000
            self.stages.append(rec)

    def to_dict(self) -> dict:
        return asdict(self)
