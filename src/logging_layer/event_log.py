"""In-memory and JSONL event logs over shared.obs.Observer. Both validate each
event before recording it; JSONLFileEventLog also appends to a .jsonl file."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from shared.obs import Observer, read_events

from .event_schema import validate_event


class InMemoryEventLog:
    def __init__(self, module: str = "rag_pipeline", *, deterministic: bool = True,
                 sink: Optional[str | Path] = None) -> None:
        self.obs = Observer(module, sink=sink, deterministic=deterministic)

    def record(self, *, module: Optional[str] = None, input: dict[str, Any],
               output: dict[str, Any], latency_ms: float,
               meta: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        ev = self.obs.emit(input=input, output=output, latency_ms=latency_ms,
                           meta=meta or {})
        if module is not None:
            ev.meta.setdefault("source_module", module)
        canonical = ev.canonical()
        validate_event(canonical)
        return canonical | {"meta": ev.meta}

    def events(self) -> list[dict[str, Any]]:
        return [e.canonical() | {"meta": e.meta} for e in self.obs.events]

    def filter(self, **meta_eq: Any) -> list[dict[str, Any]]:
        return [
            e.canonical() | {"meta": e.meta}
            for e in self.obs.events
            if all(e.meta.get(k) == v for k, v in meta_eq.items())
        ]

    def aggregate(self) -> dict[str, Any]:
        evs = self.obs.events
        return {
            "events": len(evs),
            "unique_event_ids": len({e.event_id for e in evs}),
            "modules": sorted({e.meta.get("source_module", e.module) for e in evs}),
            "total_latency_ms": round(sum(e.latency_ms for e in evs), 6),
        }

    def close(self) -> None:
        self.obs.close()


class JSONLFileEventLog(InMemoryEventLog):
    def __init__(self, path: str | Path, module: str = "rag_pipeline",
                 *, deterministic: bool = True) -> None:
        super().__init__(module, deterministic=deterministic, sink=path)
        self.path = Path(path)


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    return read_events(path)
