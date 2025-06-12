"""Core logic: RAG pipeline orchestration, tracing, retry, determinism."""
from src.abstraction.bm25_retriever import BM25Retriever
from src.abstraction.mock_generator import ExtractiveGenerator
from src.abstraction.retriever_interface import Retriever
from src.core_logic.rag_pipeline import RAGPipeline
from src.input_layer.schema import Document, Query, RetrievedDoc

DOCS = [
    Document("D1", "a credit check must complete before a sales order is confirmed"),
    Document("D2", "goods receipt increases on-hand inventory"),
]


def _pipeline():
    r = BM25Retriever().index(DOCS)
    return RAGPipeline(r, ExtractiveGenerator(), k=2)


def test_run_output_shape():
    res = _pipeline().run(Query("Q1", "credit check sales order"))
    d = res.as_dict()
    for key in ("query_id", "retrieved_docs", "retrieved_ids", "generated_answer",
                "latency_ms", "retriever", "generator"):
        assert key in d
    assert d["query_id"] == "Q1"


def test_run_accepts_plain_string():
    res = _pipeline().run("inventory")
    assert res.query_id == "adhoc"
    assert res.generated_answer


def test_tracing_records_retrieve_and_generate():
    p = _pipeline()
    p.run(Query("Q1", "credit check"))
    modules = {e["meta"]["phase"] for e in p.log.events()}
    assert {"retrieve", "generate"} <= modules


def test_determinism_three_runs():
    answers = []
    for _ in range(3):
        res = _pipeline().run(Query("Q1", "credit check sales order"))
        answers.append((res.generated_answer, tuple(d.doc_id for d in res.retrieved_docs)))
    assert answers[0] == answers[1] == answers[2]


class _FlakyRetriever(Retriever):
    name = "flaky"

    def __init__(self, fail_times: int):
        super().__init__()
        self.fail_times = fail_times
        self.calls = 0

    def _build(self):
        pass

    def _score(self, query):
        return {}

    def retrieve(self, query, k=5):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        return [RetrievedDoc("D1", 1.0, "credit check sales order", "s")]


def test_retry_recovers_from_transient_failure():
    r = _FlakyRetriever(fail_times=2)
    p = RAGPipeline(r, ExtractiveGenerator(), k=1, max_retries=3)
    res = p.run(Query("Q1", "credit"))
    assert res.retrieved_docs[0].doc_id == "D1"
    assert r.calls == 3
