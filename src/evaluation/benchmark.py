"""Run one retriever+generator over the query set: per-query records plus
aggregate means, latency percentiles and a failure-mode breakdown."""
from __future__ import annotations

import statistics
from typing import Any, Optional, Sequence

from shared.determinism import DEFAULT_SEED

from ..abstraction.generator_interface import Generator
from ..abstraction.retriever_interface import Retriever
from ..core_logic.rag_pipeline import RAGPipeline
from ..input_layer.schema import Document, Query
from ..logging_layer.event_log import InMemoryEventLog
from .failure_taxonomy import FailureMode, classify_failure
from .metrics import (
    answer_correctness_f1, context_precision_at_k, faithfulness,
    grounding_tokens, hallucination_rate,
)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def evaluate_query(pipeline: RAGPipeline, query: Query, *, k: int,
                   corpus_by_id: Optional[dict[str, Document]] = None) -> dict[str, Any]:
    result = pipeline.run(query)
    docs = result.retrieved_docs
    answer = result.generated_answer

    corpus_by_id = corpus_by_id or {}
    gold_context_texts = [
        corpus_by_id[cid].text for cid in query.expected_context_ids
        if cid in corpus_by_id
    ]
    grounding = grounding_tokens(query.gold_answer, gold_context_texts)

    mode = classify_failure(
        answer=answer,
        gold_answer=query.gold_answer,
        contexts=docs,
        expected_context_ids=query.expected_context_ids,
        grounding=grounding,
    )
    return {
        "query_id": query.query_id,
        "stress_type": query.stress_type.value,
        "retrieved_ids": [d.doc_id for d in docs],
        "answer": answer,
        "faithfulness": round(faithfulness(answer, grounding, query.gold_answer), 6),
        "hallucination_rate": round(
            hallucination_rate(answer, grounding, query.gold_answer), 6),
        "context_precision_at_k": round(
            context_precision_at_k(docs, query.expected_context_ids, k), 6),
        "answer_correctness_f1": round(
            answer_correctness_f1(answer, query.gold_answer), 6),
        "failure_mode": mode.value,
        "latency_ms": round(result.latency_ms, 6),
    }


def aggregate(per_query: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not per_query:
        return {
            "queries": 0, "faithfulness": 0.0, "hallucination_rate": 0.0,
            "context_precision_at_k": 0.0, "answer_correctness_f1": 0.0,
            "latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_p99_ms": 0.0,
            "failure_modes": {},
        }
    latencies = [r["latency_ms"] for r in per_query]
    modes: dict[str, int] = {}
    for r in per_query:
        modes[r["failure_mode"]] = modes.get(r["failure_mode"], 0) + 1
    return {
        "queries": len(per_query),
        "faithfulness": round(statistics.mean(r["faithfulness"] for r in per_query), 6),
        "hallucination_rate": round(
            statistics.mean(r["hallucination_rate"] for r in per_query), 6),
        "context_precision_at_k": round(
            statistics.mean(r["context_precision_at_k"] for r in per_query), 6),
        "answer_correctness_f1": round(
            statistics.mean(r["answer_correctness_f1"] for r in per_query), 6),
        "latency_p50_ms": round(_percentile(latencies, 50), 6),
        "latency_p95_ms": round(_percentile(latencies, 95), 6),
        "latency_p99_ms": round(_percentile(latencies, 99), 6),
        "failure_modes": {m.value: modes.get(m.value, 0) for m in FailureMode},
    }


def run_benchmark(
    retriever: Retriever,
    generator: Generator,
    queries: Sequence[Query],
    *,
    seed: int = DEFAULT_SEED,
    k: int = 5,
    corpus: Optional[Sequence[Document]] = None,
    event_log: Optional[InMemoryEventLog] = None,
) -> dict[str, Any]:
    pipeline = RAGPipeline(retriever, generator, seed=seed, k=k, event_log=event_log)
    corpus_by_id = {d.doc_id: d for d in (corpus or [])}
    per_query = [evaluate_query(pipeline, q, k=k, corpus_by_id=corpus_by_id)
                 for q in queries]
    return {
        "retriever": retriever.name,
        "generator": generator.name,
        "k": k,
        "seed": seed,
        "per_query": per_query,
        "aggregate": aggregate(per_query),
    }
