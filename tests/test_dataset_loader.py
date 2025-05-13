"""Dataset loader: the corpus parses, is deterministic, and is well-formed."""
from src.input_layer.dataset_loader import load_dataset, load_queries
from src.input_layer.stress_taxonomy import StressType


def test_corpus_loads_and_is_sorted():
    docs = load_dataset()
    assert len(docs) >= 20
    assert [d.doc_id for d in docs] == sorted(d.doc_id for d in docs)


def test_queries_load_with_all_stress_types():
    queries = load_queries()
    assert len(queries) >= 12
    seen = {q.stress_type for q in queries}
    for st in (StressType.ADVERSARIAL, StressType.AMBIGUOUS,
               StressType.LONG_CONTEXT_OVERLOAD, StressType.NORMAL):
        assert st in seen


def test_expected_context_ids_reference_real_docs():
    doc_ids = {d.doc_id for d in load_dataset()}
    for q in load_queries():
        for cid in q.expected_context_ids:
            assert cid in doc_ids, f"{q.query_id} references unknown doc {cid}"


def test_loading_is_deterministic():
    a = [d.doc_id for d in load_dataset()]
    b = [d.doc_id for d in load_dataset()]
    assert a == b
