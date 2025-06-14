"""Metrics, all pure functions. The grounding an answer is scored against is the
gold answer plus the gold context texts, not whatever got retrieved, so an
answer pulled from an off-topic passage counts as a hallucination. Degenerate
inputs return 0.0 instead of raising."""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from ..input_layer.schema import RetrievedDoc
from ..abstraction.generator_interface import ABSTAIN, split_sentences

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# dropped when checking grounding so function words don't inflate faithfulness
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "be", "for",
    "on", "by", "with", "that", "this", "it", "as", "at", "from", "into", "no",
    "not", "does", "do", "can", "must", "if", "when", "than", "then", "its",
    "based", "context", "only", "any", "every", "each", "all", "one", "more",
    "such", "so", "but", "which", "their", "they", "them", "what", "how",
})


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _content(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOPWORDS}


def _claims(answer: str) -> list[str]:
    return [s for s in split_sentences(answer) if s.strip()]


def grounding_tokens(gold_answer: str, grounding_texts: Iterable[str]) -> set[str]:
    """Content tokens of the gold answer plus the gold context texts."""
    blob = _content(gold_answer)
    for t in grounding_texts:
        blob |= _content(t)
    return blob


def _is_grounded(claim: str, grounding: set[str], *, min_ratio: float = 0.6) -> bool:
    claim_tokens = _content(claim)
    if not claim_tokens:
        return True  # punctuation-only claim, treat as grounded
    overlap = len(claim_tokens & grounding)
    return (overlap / len(claim_tokens)) >= min_ratio


def faithfulness(answer: str, grounding: set[str] | Sequence[RetrievedDoc],
                 gold_answer: str = "") -> float:
    """Fraction of claims grounded. grounding may be a token set or RetrievedDocs
    (then doc texts + gold_answer form the grounding)."""
    if answer.strip() == ABSTAIN or not answer.strip():
        return 0.0
    g = _as_grounding(grounding, gold_answer)
    claims = _claims(answer)
    if not claims:
        return 0.0
    grounded = sum(1 for c in claims if _is_grounded(c, g))
    return grounded / len(claims)


def hallucination_rate(answer: str, grounding: set[str] | Sequence[RetrievedDoc],
                       gold_answer: str = "") -> float:
    """Fraction of claims not grounded. Abstention scores 0.0."""
    if answer.strip() == ABSTAIN or not answer.strip():
        return 0.0
    g = _as_grounding(grounding, gold_answer)
    claims = _claims(answer)
    if not claims:
        return 0.0
    ungrounded = sum(1 for c in claims if not _is_grounded(c, g))
    return ungrounded / len(claims)


def _as_grounding(grounding: set[str] | Sequence[RetrievedDoc],
                  gold_answer: str) -> set[str]:
    if isinstance(grounding, set):
        return grounding
    return grounding_tokens(gold_answer, (d.text for d in grounding))


def context_precision_at_k(retrieved: Sequence[RetrievedDoc],
                           relevant_ids: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    relevant = set(relevant_ids)
    hits = sum(1 for d in top if d.doc_id in relevant)
    return hits / k


def answer_correctness_f1(answer: str, gold: str) -> float:
    """Token-overlap F1 against the gold answer."""
    if not answer.strip() or not gold.strip() or answer.strip() == ABSTAIN:
        return 0.0
    a = _content(answer)
    g = _content(gold)
    if not a or not g:
        return 0.0
    tp = len(a & g)
    if tp == 0:
        return 0.0
    precision = tp / len(a)
    recall = tp / len(g)
    return 2 * precision * recall / (precision + recall)
