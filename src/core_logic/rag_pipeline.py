"""The pipeline: retrieve then generate, tracing each stage. Retrieval gets a
small retry wrapper so a transient failure doesn't kill the run. Seeding happens
per run() so repeats at the same seed match."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from shared.determinism import DEFAULT_SEED, set_seed

from ..abstraction.generator_interface import Generator
from ..abstraction.retriever_interface import Retriever
from ..input_layer.schema import Query, RetrievedDoc
from ..logging_layer.event_log import InMemoryEventLog


@dataclass
class PipelineResult:
    query_id: str
    query_text: str
    retrieved_docs: list[RetrievedDoc] = field(default_factory=list)
    generated_answer: str = ""
    latency_ms: float = 0.0
    retriever: str = ""
    generator: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "retrieved_docs": [d.as_dict() for d in self.retrieved_docs],
            "retrieved_ids": [d.doc_id for d in self.retrieved_docs],
            "generated_answer": self.generated_answer,
            "latency_ms": round(self.latency_ms, 6),
            "retriever": self.retriever,
            "generator": self.generator,
        }


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator,
                 *, seed: int = DEFAULT_SEED, k: int = 5, max_retries: int = 3,
                 event_log: Optional[InMemoryEventLog] = None) -> None:
        self.retriever = retriever
        self.generator = generator
        self.seed = seed
        self.k = k
        self.max_retries = max(1, max_retries)
        self.log = event_log if event_log is not None else InMemoryEventLog(
            "rag_pipeline", deterministic=True)

    def _retrieve_with_retry(self, query_text: str) -> list[RetrievedDoc]:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self.retriever.retrieve(query_text, k=self.k)
            except Exception as exc:  # pragma: no cover - exercised via test stub
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        return []

    def run(self, query: Query | str) -> PipelineResult:
        set_seed(self.seed)
        if isinstance(query, Query):
            query_id, query_text = query.query_id, query.text
        else:
            query_id, query_text = "adhoc", str(query)

        start = time.perf_counter()

        docs = self._retrieve_with_retry(query_text)
        self.log.record(
            module="retriever",
            input={"query_id": query_id, "query": query_text, "k": self.k},
            output={"retrieved_ids": [d.doc_id for d in docs]},
            latency_ms=0.0,
            meta={"source_module": "retriever", "phase": "retrieve",
                  "retriever": self.retriever.name, "query_id": query_id},
        )

        answer = self.generator.generate(query_text, docs, seed=self.seed)
        self.log.record(
            module="generator",
            input={"query_id": query_id, "query": query_text,
                   "context_ids": [d.doc_id for d in docs]},
            output={"answer": answer},
            latency_ms=0.0,
            meta={"source_module": "generator", "phase": "generate",
                  "generator": self.generator.name, "query_id": query_id},
        )

        latency_ms = (time.perf_counter() - start) * 1000.0
        return PipelineResult(
            query_id=query_id,
            query_text=query_text,
            retrieved_docs=docs,
            generated_answer=answer,
            latency_ms=latency_ms,
            retriever=self.retriever.name,
            generator=self.generator.name,
        )
