"""Okapi BM25, hand-rolled so there's no rank_bm25 dependency."""
from __future__ import annotations

import math
from collections import Counter

from shared.determinism import DEFAULT_SEED

from .retriever_interface import Retriever, tokenize


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75,
                 seed: int = DEFAULT_SEED) -> None:
        super().__init__(seed=seed)
        self.k1 = k1
        self.b = b
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_len: dict[str, int] = {}
        self._tf: dict[str, Counter] = {}
        self._df: Counter = Counter()
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._n: int = 0

    def _build(self) -> None:
        self._doc_tokens = {d.doc_id: tokenize(d.text) for d in self._docs}
        self._doc_len = {did: len(toks) for did, toks in self._doc_tokens.items()}
        self._tf = {did: Counter(toks) for did, toks in self._doc_tokens.items()}
        self._n = len(self._docs)
        self._avgdl = (sum(self._doc_len.values()) / self._n) if self._n else 0.0
        self._df = Counter()
        for toks in self._doc_tokens.values():
            for term in set(toks):
                self._df[term] += 1
        self._idf = {
            term: math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }

    def _score(self, query: str) -> dict[str, float]:
        q_terms = tokenize(query)
        scores: dict[str, float] = {d.doc_id: 0.0 for d in self._docs}
        if not q_terms or self._avgdl == 0.0:
            return scores
        for did in scores:
            tf = self._tf[did]
            dl = self._doc_len[did]
            denom_norm = self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                s += idf * (f * (self.k1 + 1.0)) / (f + denom_norm)
            scores[did] = s
        return scores
