"""Dump a query's trace to a single static HTML file (no JS, no assets) so you
can eyeball what the retriever and generator did per stage."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence


def _row(event: dict[str, Any]) -> str:
    meta = event.get("meta", {})
    return (
        "<tr>"
        f"<td>{html.escape(str(event.get('event_id', '')))}</td>"
        f"<td>{html.escape(str(meta.get('source_module', event.get('module', ''))))}</td>"
        f"<td>{html.escape(str(meta.get('phase', '')))}</td>"
        f"<td>{event.get('latency_ms', 0)}</td>"
        f"<td><pre>{html.escape(json.dumps(event.get('input', {}), sort_keys=True, indent=2))}</pre></td>"
        f"<td><pre>{html.escape(json.dumps(event.get('output', {}), sort_keys=True, indent=2))}</pre></td>"
        "</tr>"
    )


def render_query_trace_html(query_id: str, events: Sequence[dict[str, Any]]) -> str:
    rows = "\n".join(_row(e) for e in events)
    return (
        "<!doctype html>\n"
        "<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>Trace {html.escape(query_id)}</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:1.5rem;background:#0f1115;color:#e6e6e6}"
        "h1{font-size:1.2rem}"
        "table{border-collapse:collapse;width:100%;font-size:.85rem}"
        "th,td{border:1px solid #333;padding:.4rem;text-align:left;vertical-align:top}"
        "th{background:#1b1f27}"
        "pre{margin:0;white-space:pre-wrap;max-width:34rem}"
        "</style></head><body>"
        f"<h1>RAG query trace: {html.escape(query_id)}</h1>"
        f"<p>{len(events)} event(s)</p>"
        "<table><thead><tr>"
        "<th>event_id</th><th>module</th><th>phase</th><th>latency_ms</th>"
        "<th>input</th><th>output</th>"
        "</tr></thead><tbody>"
        f"{rows}"
        "</tbody></table></body></html>"
    )


def export_query_trace_html(query_id: str, events: Sequence[dict[str, Any]],
                            path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_query_trace_html(query_id, events), encoding="utf-8")
    return p
