"""Validate a canonical trace event. Extras beyond REQUIRED_FIELDS live in meta."""
from __future__ import annotations

from typing import Any

REQUIRED_FIELDS: tuple[str, ...] = (
    "event_id", "timestamp", "module", "input", "output", "latency_ms",
)


class EventSchemaError(ValueError):
    pass


def validate_event(event: dict[str, Any]) -> dict[str, Any]:
    for f in REQUIRED_FIELDS:
        if f not in event:
            raise EventSchemaError(f"event missing required field: {f}")
    if not isinstance(event["event_id"], str) or not event["event_id"]:
        raise EventSchemaError("event_id must be a non-empty string")
    if not isinstance(event["module"], str) or not event["module"]:
        raise EventSchemaError("module must be a non-empty string")
    if not isinstance(event["input"], dict):
        raise EventSchemaError("input must be a dict")
    if not isinstance(event["output"], dict):
        raise EventSchemaError("output must be a dict")
    if not isinstance(event["timestamp"], (int, float)):
        raise EventSchemaError("timestamp must be numeric")
    if not isinstance(event["latency_ms"], (int, float)) or event["latency_ms"] < 0:
        raise EventSchemaError("latency_ms must be a non-negative number")
    return event
