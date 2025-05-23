"""BM25 retriever tests."""
from src.abstraction.bm25_retriever import BM25Retriever
from src.input_layer.schema import Document

DOCS = [
    Document("D1", "credit check before a sales order is confirmed"),
    Document("D2", "goods receipt increases on-hand inventory"),
    Document("D3", "three way matching approves the supplier invoice"),
]


def _r():
    return BM25Retriever().index(DOCS)


def test_retrieve_ranks_relevant_first():
    hits = _r().retrieve("credit check sales order", k=3)
    assert hits[0].doc_id == "D1"


def test_topk_respected():
    assert len(_r().retrieve("inventory", k=1)) == 1


def test_empty_query_returns_zero_scores():
    hits = _r().retrieve("", k=3)
    assert all(h.score == 0.0 for h in hits)


def test_deterministic_tie_break_by_doc_id():
    # A query with no matching terms -> all-zero scores, tie-broken by doc_id.
    hits = _r().retrieve("zzz nonexistent", k=3)
    assert [h.doc_id for h in hits] == ["D1", "D2", "D3"]
