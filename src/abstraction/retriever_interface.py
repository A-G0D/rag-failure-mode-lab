"""Shared retriever contract: index a corpus, then retrieve(query, k).

Ties are broken by doc_id so a fixed seed gives reproducible output.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Sequence

from shared.determinism import DEFAULT_SEED

from ..input_layer.schema import Document, RetrievedDoc

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Retriever(ABC):
    name: str = "retriever"

    def __init__(self, *, seed: int = DEFAULT_SEED) -> None:
        self.seed = seed
        self._docs: list[Document] = []
        self._by_id: dict[str, Document] = {}

    def index(self, documents: Sequence[Document]) -> "Retriever":
        self._docs = sorted(documents, key=lambda d: d.doc_id)
        self._by_id = {d.doc_id: d for d in self._docs}
        self._build()
        return self

    @abstractmethod
    def _build(self) -> None:
        """Build the index structures from self._docs."""

    @abstractmethod
    def _score(self, query: str) -> dict[str, float]:
        """{doc_id: score} for the query, higher is better."""

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        if k <= 0 or not self._docs:
            return []
        scores = self._score(query)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        out: list[RetrievedDoc] = []
        for doc_id, score in ranked[:k]:
            doc = self._by_id[doc_id]
            out.append(RetrievedDoc(doc_id=doc_id, score=float(score),
                                    text=doc.text, source=doc.source))
        return out

    def __len__(self) -> int:
        return len(self._docs)
