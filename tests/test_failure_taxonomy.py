"""Failure taxonomy classification tests (deterministic, no LLM judge)."""
from src.abstraction.generator_interface import ABSTAIN
from src.evaluation.failure_taxonomy import FailureMode, classify_failure
from src.evaluation.metrics import grounding_tokens
from src.input_layer.schema import RetrievedDoc

GOLD = "a credit check must complete before a sales order is confirmed"
CTX_GOOD = [RetrievedDoc("D1", 1.0, GOLD, "s")]
CTX_BAD = [RetrievedDoc("D9", 1.0, "goods receipt increases on-hand inventory", "s")]


def _g(gold, texts):
    return grounding_tokens(gold, texts)


def test_correct():
    mode = classify_failure(
        answer="A credit check must complete before a sales order is confirmed.",
        gold_answer=GOLD, contexts=CTX_GOOD, expected_context_ids=["D1"],
        grounding=_g(GOLD, [GOLD]))
    assert mode == FailureMode.CORRECT


def test_hallucination():
    mode = classify_failure(
        answer="The moon is made of green cheese and supplier invoices.",
        gold_answer=GOLD, contexts=CTX_GOOD, expected_context_ids=["D1"],
        grounding=_g(GOLD, [GOLD]))
    assert mode == FailureMode.HALLUCINATION


def test_irrelevant_retrieval():
    mode = classify_failure(
        answer="Goods receipt increases on-hand inventory.",
        gold_answer=GOLD, contexts=CTX_BAD, expected_context_ids=["D1"],
        grounding=_g(GOLD, [GOLD]))
    # Expected D1 not retrieved (only D9), and answer matches grounding poorly.
    assert mode in (FailureMode.IRRELEVANT_RETRIEVAL, FailureMode.HALLUCINATION)


def test_context_mismatch_on_abstention():
    mode = classify_failure(
        answer=ABSTAIN, gold_answer=GOLD, contexts=CTX_GOOD,
        expected_context_ids=["D1"], grounding=_g(GOLD, [GOLD]))
    assert mode == FailureMode.CONTEXT_MISMATCH


def test_partial_correctness():
    mode = classify_failure(
        answer="A credit check is required.",  # grounded, low overlap with gold
        gold_answer=GOLD, contexts=CTX_GOOD, expected_context_ids=["D1"],
        grounding=_g(GOLD, [GOLD]))
    assert mode in (FailureMode.PARTIAL_CORRECTNESS, FailureMode.CORRECT)
