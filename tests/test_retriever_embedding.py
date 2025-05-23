"""Embedding retriever (TF-IDF + cosine) tests."""
from src.abstraction.embedding_retriever import EmbeddingRetriever
from src.input_layer.schema import Document

DOCS = [
    Document("D1", "purchase order is a binding commitment to a supplier"),
    Document("D2", "cycle count audits a subset of inventory locations"),
    Document("D3", "invoice total equals shipped line amounts plus tax"),
]


def _r():
    return EmbeddingRetriever().index(DOCS)


def test_retrieve_relevant_doc():
    hits = _r().retrieve("inventory cycle count locations", k=3)
    assert hits[0].doc_id == "D2"


def test_scores_are_cosine_bounded():
    for h in _r().retrieve("purchase order supplier", k=3):
        assert -1.0001 <= h.score <= 1.0001


def test_reproducible():
    a = [(h.doc_id, round(h.score, 6)) for h in _r().retrieve("supplier invoice", k=3)]
    b = [(h.doc_id, round(h.score, 6)) for h in _r().retrieve("supplier invoice", k=3)]
    assert a == b
