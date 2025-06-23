"""Run every retriever x generator combo over the query set, plus per-stress-
type slices. Feeds the comparative report."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from shared.determinism import DEFAULT_SEED

from ..abstraction.bm25_retriever import BM25Retriever
from ..abstraction.embedding_retriever import EmbeddingRetriever
from ..abstraction.hybrid_retriever import HybridRetriever
from ..abstraction.mock_generator import ExtractiveGenerator
from ..abstraction.template_generator import TemplateGenerator
from ..input_layer.schema import Document, Query
from .benchmark import aggregate, run_benchmark


def build_retrievers(documents: Sequence[Document], *, seed: int = DEFAULT_SEED,
                     hybrid_alpha: float = 0.5, k1: float = 1.5, b: float = 0.75) -> dict:
    retrievers = {
        "bm25": BM25Retriever(k1=k1, b=b, seed=seed),
        "embedding": EmbeddingRetriever(seed=seed),
        "hybrid": HybridRetriever(alpha=hybrid_alpha, k1=k1, b=b, seed=seed),
    }
    for r in retrievers.values():
        r.index(documents)
    return retrievers


def build_generators() -> dict:
    return {
        "extractive": ExtractiveGenerator(),
        "template": TemplateGenerator(),
    }


def _by_stress_type(per_query: Sequence[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for r in per_query:
        buckets.setdefault(r["stress_type"], []).append(r)
    return {stype: aggregate(rows) for stype, rows in sorted(buckets.items())}


def run_stress_suite(
    documents: Sequence[Document],
    queries: Sequence[Query],
    *,
    seed: int = DEFAULT_SEED,
    k: int = 5,
    hybrid_alpha: float = 0.5,
    k1: float = 1.5,
    b: float = 0.75,
    combos: Optional[Sequence[tuple[str, str]]] = None,
) -> dict[str, Any]:
    """Keyed "<retriever>+<generator>" -> benchmark result with a per-stress slice."""
    retrievers = build_retrievers(documents, seed=seed, hybrid_alpha=hybrid_alpha,
                                  k1=k1, b=b)
    generators = build_generators()
    if combos is None:
        combos = [(rn, gn) for rn in retrievers for gn in generators]

    results: dict[str, Any] = {}
    for rname, gname in combos:
        bench = run_benchmark(retrievers[rname], generators[gname], queries,
                              seed=seed, k=k, corpus=documents)
        bench["by_stress_type"] = _by_stress_type(bench["per_query"])
        results[f"{rname}+{gname}"] = bench
    return results
