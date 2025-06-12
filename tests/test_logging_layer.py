"""Logging layer: event schema, event log, JSONL roundtrip, HTML export."""
import json

import pytest

from src.logging_layer.debug_hooks import (
    export_query_trace_html, render_query_trace_html,
)
from src.logging_layer.event_log import InMemoryEventLog, JSONLFileEventLog, load_trace
from src.logging_layer.event_schema import (
    EventSchemaError, REQUIRED_FIELDS, validate_event,
)


def _valid_event():
    return {"event_id": "rag_pipeline-000001", "timestamp": 0.001,
            "module": "rag_pipeline", "input": {}, "output": {}, "latency_ms": 1.0}


def test_validate_event_accepts_valid():
    assert validate_event(_valid_event())


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_validate_event_rejects_missing_field(field):
    ev = _valid_event()
    del ev[field]
    with pytest.raises(EventSchemaError):
        validate_event(ev)


def test_validate_event_rejects_negative_latency():
    ev = _valid_event()
    ev["latency_ms"] = -1.0
    with pytest.raises(EventSchemaError):
        validate_event(ev)


def test_in_memory_log_records_and_filters():
    log = InMemoryEventLog("rag_pipeline", deterministic=True)
    log.record(module="retriever", input={"q": "x"}, output={"ids": []},
               latency_ms=0.0, meta={"source_module": "retriever", "phase": "retrieve"})
    log.record(module="generator", input={"q": "x"}, output={"answer": "y"},
               latency_ms=0.0, meta={"source_module": "generator", "phase": "generate"})
    assert len(log.events()) == 2
    assert len(log.filter(phase="retrieve")) == 1
    agg = log.aggregate()
    assert agg["events"] == 2 and agg["unique_event_ids"] == 2


def test_jsonl_file_log_roundtrip(tmp_path):
    path = tmp_path / "logs" / "trace.jsonl"
    log = JSONLFileEventLog(path, "rag_pipeline", deterministic=True)
    log.record(input={"q": "x"}, output={"a": "y"}, latency_ms=1.0)
    log.close()
    rows = load_trace(path)
    assert len(rows) == 1
    for line in path.read_text(encoding="utf-8").splitlines():
        json.loads(line)  # valid JSON


def test_html_export_is_self_contained(tmp_path):
    events = [{"event_id": "e1", "timestamp": 0.001, "module": "retriever",
               "input": {"q": "x"}, "output": {"ids": ["D1"]}, "latency_ms": 1.0,
               "meta": {"source_module": "retriever", "phase": "retrieve"}}]
    html = render_query_trace_html("Q1", events)
    assert "<!doctype html>" in html
    assert "<script" not in html.lower()  # no JS -> no external dependency
    p = export_query_trace_html("Q1", events, tmp_path / "q.html")
    assert p.exists()
