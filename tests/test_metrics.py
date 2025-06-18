"""Metric correctness tests (faithfulness, precision@k, hallucination, F1)."""
from src.abstraction.generator_interface import ABSTAIN
from src.evaluation.metrics import (
    answer_correctness_f1, context_precision_at_k, faithfulness,
    grounding_tokens, hallucination_rate,
)
from src.input_layer.schema import RetrievedDoc

GOLD = "a credit check must complete before a sales order is confirmed"
G = grounding_tokens(GOLD, [GOLD])


def test_faithfulness_full_when_grounded():
    assert faithfulness("A credit check must complete before a sales order.", G) == 1.0


def test_faithfulness_zero_on_abstain():
    assert faithfulness(ABSTAIN, G) == 0.0


def test_hallucination_full_when_ungrounded():
    assert hallucination_rate("Bananas orbit the supplier moon.", G) == 1.0


def test_hallucination_zero_on_abstain():
    assert hallucination_rate(ABSTAIN, G) == 0.0


def test_faithfulness_plus_hallucination_complement():
    ans = "A credit check must complete. Bananas orbit the moon."
    f = faithfulness(ans, G)
    h = hallucination_rate(ans, G)
    assert abs((f + h) - 1.0) < 1e-9


def test_context_precision_at_k():
    docs = [RetrievedDoc("D1", 1.0, "x"), RetrievedDoc("D2", 0.5, "y"),
            RetrievedDoc("D3", 0.1, "z")]
    assert context_precision_at_k(docs, ["D1", "D3"], 2) == 0.5  # D1 in top-2
    assert context_precision_at_k(docs, [], 2) == 0.0
    assert context_precision_at_k(docs, ["D1"], 0) == 0.0


def test_answer_correctness_f1_bounds():
    assert answer_correctness_f1(GOLD, GOLD) == 1.0
    assert answer_correctness_f1("", GOLD) == 0.0
    assert answer_correctness_f1("bananas moon", GOLD) == 0.0
    f1 = answer_correctness_f1("a credit check is needed", GOLD)
    assert 0.0 < f1 < 1.0


def test_metrics_handle_empty_inputs():
    assert faithfulness("", G) == 0.0
    assert hallucination_rate("", G) == 0.0
    assert answer_correctness_f1("x", "") == 0.0
