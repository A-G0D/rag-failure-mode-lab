"""Hybrid retriever: min-max normalize BM25 and embedding scores, then blend
them as alpha*embedding + (1-alpha)*bm25. alpha=0 is pure BM25, alpha=1 pure
embedding."""
from __future__ import annotations

from typing import Sequence

from shared.determinism import DEFAULT_SEED

from ..input_layer.schema import Document
from .retriever_interface import Retriever
from .bm25_retriever import BM25Retriever
from .embedding_retriever import EmbeddingRetriever


def _min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi <= lo:
        return {k: 0.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, *, alpha: float = 0.5, k1: float = 1.5, b: float = 0.75,
                 seed: int = DEFAULT_SEED) -> None:
        super().__init__(seed=seed)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self._bm25 = BM25Retriever(k1=k1, b=b, seed=seed)
        self._emb = EmbeddingRetriever(seed=seed)

    def index(self, documents: Sequence[Document]) -> "HybridRetriever":
        super().index(documents)
        return self

    def _build(self) -> None:
        self._bm25.index(self._docs)
        self._emb.index(self._docs)

    def _score(self, query: str) -> dict[str, float]:
        bm = _min_max(self._bm25._score(query))
        em = _min_max(self._emb._score(query))
        ids = set(bm) | set(em)
        return {
            did: self.alpha * em.get(did, 0.0) + (1.0 - self.alpha) * bm.get(did, 0.0)
            for did in ids
        }
