"""Benchmark + aggregation tests."""
from src.abstraction.bm25_retriever import BM25Retriever
from src.abstraction.mock_generator import ExtractiveGenerator
from src.evaluation.benchmark import aggregate, run_benchmark
from src.input_layer.dataset_loader import load_dataset, load_queries


def test_run_benchmark_shape():
    docs = load_dataset()
    queries = load_queries()
    r = BM25Retriever().index(docs)
    bench = run_benchmark(r, ExtractiveGenerator(), queries, k=5, corpus=docs)
    assert bench["retriever"] == "bm25"
    assert bench["generator"] == "extractive"
    assert len(bench["per_query"]) == len(queries)
    agg = bench["aggregate"]
    for key in ("faithfulness", "hallucination_rate", "context_precision_at_k",
                "answer_correctness_f1", "latency_p50_ms", "failure_modes"):
        assert key in agg


def test_aggregate_empty():
    agg = aggregate([])
    assert agg["queries"] == 0
    assert agg["faithfulness"] == 0.0


def test_latency_percentiles_monotonic():
    docs = load_dataset()
    queries = load_queries()
    r = BM25Retriever().index(docs)
    agg = run_benchmark(r, ExtractiveGenerator(), queries, corpus=docs)["aggregate"]
    assert agg["latency_p50_ms"] <= agg["latency_p95_ms"] <= agg["latency_p99_ms"]


def test_metrics_in_unit_range():
    docs = load_dataset()
    queries = load_queries()
    r = BM25Retriever().index(docs)
    agg = run_benchmark(r, ExtractiveGenerator(), queries, corpus=docs)["aggregate"]
    for m in ("faithfulness", "hallucination_rate", "context_precision_at_k",
              "answer_correctness_f1"):
        assert 0.0 <= agg[m] <= 1.0
