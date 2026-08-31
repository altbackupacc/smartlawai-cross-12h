"""PipelineTrace behavior: stage timing/status, exception re-raise without
silently dropping the record, and JSON round-tripping."""
from __future__ import annotations

import json


def test_stage_records_timing_and_status():
    from smartlawai.trace import PipelineTrace

    trace = PipelineTrace(trace_id="tr-1", scope=None)
    with trace.stage("ingest") as rec:
        rec.detail = {"n": 1}

    assert len(trace.stages) == 1
    assert trace.stages[0].name == "ingest"
    assert trace.stages[0].status == "ok"
    assert trace.stages[0].duration_ms >= 0
    assert trace.stages[0].detail == {"n": 1}


def test_stage_records_error_and_reraises():
    from smartlawai.trace import PipelineTrace

    trace = PipelineTrace(trace_id="tr-2", scope=None)
    try:
        with trace.stage("retrieve"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError to propagate")

    assert len(trace.stages) == 1
    assert trace.stages[0].status == "error"
    assert trace.stages[0].error == "boom"


def test_to_dict_round_trips_through_json():
    from smartlawai.scope import Scope
    from smartlawai.trace import PipelineTrace

    trace = PipelineTrace(trace_id="tr-3", scope=Scope(doc_ids=("d1",), owner_id="alice"))
    with trace.stage("verify") as rec:
        rec.detail = {"score": 0.9}

    payload = json.loads(json.dumps(trace.to_dict(), default=str))
    assert payload["trace_id"] == "tr-3"
    assert payload["stages"][0]["name"] == "verify"
    assert payload["scope"]["owner_id"] == "alice"
