"""Bucket an answer into a failure mode from the metrics + retrieval, no LLM
judge. The precedence order is in classify_failure."""
from __future__ import annotations

from enum import Enum
from typing import Sequence

from ..input_layer.schema import RetrievedDoc
from ..abstraction.generator_interface import ABSTAIN
from .metrics import answer_correctness_f1, faithfulness, hallucination_rate


class FailureMode(str, Enum):
    CORRECT = "correct"
    HALLUCINATION = "hallucination"
    PARTIAL_CORRECTNESS = "partial_correctness"
    IRRELEVANT_RETRIEVAL = "irrelevant_retrieval"
    CONTEXT_MISMATCH = "context_mismatch"


def classify_failure(
    *,
    answer: str,
    gold_answer: str,
    contexts: Sequence[RetrievedDoc],
    expected_context_ids: Sequence[str],
    grounding: set[str] | None = None,
    correctness_threshold: float = 0.5,
    hallucination_threshold: float = 0.34,
    faithfulness_threshold: float = 0.5,
) -> FailureMode:
    retrieved_ids = {d.doc_id for d in contexts}
    expected = set(expected_context_ids)
    retrieval_ok = (not expected) or bool(retrieved_ids & expected)

    g = grounding if grounding is not None else None
    f1 = answer_correctness_f1(answer, gold_answer)
    halluc = (hallucination_rate(answer, g, gold_answer) if g is not None
              else hallucination_rate(answer, contexts, gold_answer))
    faith = (faithfulness(answer, g, gold_answer) if g is not None
             else faithfulness(answer, contexts, gold_answer))
    is_abstain = answer.strip() == ABSTAIN or not answer.strip()

    if f1 >= correctness_threshold and halluc <= hallucination_threshold and not is_abstain:
        return FailureMode.CORRECT
    if halluc > hallucination_threshold:
        return FailureMode.HALLUCINATION
    if not retrieval_ok:
        return FailureMode.IRRELEVANT_RETRIEVAL
    # context was there but the answer abstained or barely used it
    if is_abstain or faith < faithfulness_threshold:
        return FailureMode.CONTEXT_MISMATCH
    return FailureMode.PARTIAL_CORRECTNESS
