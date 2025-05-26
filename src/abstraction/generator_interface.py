"""Generator contract. Both generators only emit text drawn from the contexts;
with no context they return ABSTAIN, which the taxonomy counts as a non-answer
rather than a hallucination."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Sequence

from shared.determinism import DEFAULT_SEED

from ..input_layer.schema import RetrievedDoc

ABSTAIN = "I cannot answer this from the provided context."

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def content_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Generator(ABC):
    name: str = "generator"

    @abstractmethod
    def generate(self, query: str, contexts: Sequence[RetrievedDoc],
                 *, seed: int = DEFAULT_SEED) -> str:
        ...

    def __call__(self, query: str, contexts: Sequence[RetrievedDoc],
                 *, seed: int = DEFAULT_SEED) -> str:
        return self.generate(query, contexts, seed=seed)
