"""Logging layer: structured event schema, event log, debug hooks."""
from .event_schema import EventSchemaError, validate_event, REQUIRED_FIELDS
from .event_log import InMemoryEventLog, JSONLFileEventLog
from .debug_hooks import export_query_trace_html

__all__ = [
    "EventSchemaError", "validate_event", "REQUIRED_FIELDS",
    "InMemoryEventLog", "JSONLFileEventLog", "export_query_trace_html",
]
