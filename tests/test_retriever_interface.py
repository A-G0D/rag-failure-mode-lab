"""Retriever interface contract: ABC compliance + result schema + seeding."""
import pytest

from src.abstraction.bm25_retriever import BM25Retriever
from src.abstraction.embedding_retriever import EmbeddingRetriever
from src.abstraction.hybrid_retriever import HybridRetriever
from src.abstraction.retriever_interface import Retriever, tokenize
from src.input_layer.schema import Document

DOCS = [Document(f"D{i}", f"document number {i} about inventory and orders")
        for i in range(1, 6)]


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        Retriever()  # type: ignore[abstract]


@pytest.mark.parametrize("ctor", [
    lambda: BM25Retriever(seed=1),
    lambda: EmbeddingRetriever(seed=1),
    lambda: HybridRetriever(alpha=0.5, seed=1),
])
def test_result_schema(ctor):
    r = ctor().index(DOCS)
    hits = r.retrieve("inventory orders", k=3)
    assert len(hits) == 3
    for h in hits:
        assert isinstance(h.doc_id, str) and h.doc_id
        assert isinstance(h.score, float)
        assert isinstance(h.text, str)


def test_k_zero_returns_empty():
    assert BM25Retriever().index(DOCS).retrieve("x", k=0) == []


def test_tokenize_lowercases():
    assert tokenize("Hello WORLD 123!") == ["hello", "world", "123"]


@pytest.mark.parametrize("ctor", [
    lambda: BM25Retriever(seed=7),
    lambda: EmbeddingRetriever(seed=7),
    lambda: HybridRetriever(alpha=0.4, seed=7),
])
def test_two_instances_identical(ctor):
    a = ctor().index(DOCS).retrieve("inventory orders", 5)
    b = ctor().index(DOCS).retrieve("inventory orders", 5)
    assert [(h.doc_id, round(h.score, 6)) for h in a] == \
           [(h.doc_id, round(h.score, 6)) for h in b]
