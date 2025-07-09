"""End to end: a couple of combos run clean and the trace log is valid JSONL."""
import json

from src.abstraction.embedding_retriever import EmbeddingRetriever
from src.abstraction.hybrid_retriever import HybridRetriever
from src.abstraction.mock_generator import ExtractiveGenerator
from src.abstraction.template_generator import TemplateGenerator
from src.core_logic.rag_pipeline import RAGPipeline
from src.input_layer.dataset_loader import load_dataset, load_queries
from src.input_layer.stress_taxonomy import StressType
from src.logging_layer.event_log import JSONLFileEventLog, load_trace


def test_adversarial_run_produces_answers_and_trace(tmp_path):
    docs = load_dataset()
    queries = [q for q in load_queries() if q.stress_type == StressType.ADVERSARIAL]
    retriever = HybridRetriever(alpha=0.5).index(docs)
    log = JSONLFileEventLog(tmp_path / "run.jsonl", "rag_pipeline", deterministic=True)
    pipeline = RAGPipeline(retriever, TemplateGenerator(), k=5, event_log=log)
    results = [pipeline.run(q) for q in queries]
    log.close()
    assert results and all(r.generated_answer for r in results)
    rows = load_trace(tmp_path / "run.jsonl")
    assert len(rows) == 2 * len(queries)  # retrieve + generate per query


def test_combos_dont_crash():
    docs = load_dataset()
    queries = load_queries()
    combos = [
        (EmbeddingRetriever(), ExtractiveGenerator()),
        (HybridRetriever(alpha=0.5), TemplateGenerator()),
    ]
    for retriever, generator in combos:
        retriever.index(docs)
        p = RAGPipeline(retriever, generator, k=5)
        for q in queries:
            assert isinstance(p.run(q).generated_answer, str)


def test_log_file_is_valid_jsonl(tmp_path):
    docs = load_dataset()
    log = JSONLFileEventLog(tmp_path / "t.jsonl", deterministic=True)
    p = RAGPipeline(HybridRetriever().index(docs), TemplateGenerator(),
                    k=5, event_log=log)
    p.run(load_queries()[0])
    log.close()
    for line in (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines():
        json.loads(line)
