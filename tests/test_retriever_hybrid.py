"""Hybrid retriever (BM25 + embedding fusion) tests."""
import pytest

from src.abstraction.hybrid_retriever import HybridRetriever
from src.input_layer.schema import Document

DOCS = [
    Document("D1", "credit check before a sales order is confirmed"),
    Document("D2", "goods receipt increases on-hand inventory for the warehouse"),
    Document("D3", "three way matching approves the supplier invoice for payment"),
]


def test_alpha_bounds_validated():
    with pytest.raises(ValueError):
        HybridRetriever(alpha=1.5)


def test_alpha_zero_matches_bm25_ranking():
    from src.abstraction.bm25_retriever import BM25Retriever
    hybrid = HybridRetriever(alpha=0.0).index(DOCS)
    bm25 = BM25Retriever().index(DOCS)
    q = "supplier invoice payment"
    assert [h.doc_id for h in hybrid.retrieve(q, 3)] == \
           [h.doc_id for h in bm25.retrieve(q, 3)]


def test_hybrid_returns_topk():
    r = HybridRetriever(alpha=0.5).index(DOCS)
    assert len(r.retrieve("inventory warehouse", 2)) == 2


def test_reproducible():
    r = HybridRetriever(alpha=0.5).index(DOCS)
    a = [(h.doc_id, round(h.score, 6)) for h in r.retrieve("credit sales order", 3)]
    b = [(h.doc_id, round(h.score, 6)) for h in r.retrieve("credit sales order", 3)]
    assert a == b
