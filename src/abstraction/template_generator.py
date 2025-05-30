"""Improved generator. Two changes over the extractive baseline cut the
hallucination rate: it keeps only sentences within a margin of the best
overlap (tangential ones get dropped), and it abstains outright when the best
sentence doesn't clear a minimum overlap, instead of guessing."""
from __future__ import annotations

from typing import Sequence

from ..input_layer.schema import RetrievedDoc
from .generator_interface import (
    ABSTAIN, DEFAULT_SEED, Generator, content_tokens, split_sentences)


class TemplateGenerator(Generator):
    name = "template"

    def __init__(self, *, max_sentences: int = 2, min_overlap: int = 2,
                 margin: int = 1) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap
        self.margin = margin

    def generate(self, query: str, contexts: Sequence[RetrievedDoc],
                 *, seed: int = DEFAULT_SEED) -> str:
        if not contexts:
            return ABSTAIN
        q_tokens = set(content_tokens(query))

        scored: list[tuple[int, int, str]] = []
        for rank, ctx in enumerate(contexts):
            for sentence in split_sentences(ctx.text):
                overlap = len(q_tokens & set(content_tokens(sentence)))
                scored.append((overlap, rank, sentence))

        if not scored:
            return ABSTAIN

        best_overlap = max(o for o, _, _ in scored)
        if best_overlap < self.min_overlap:
            return ABSTAIN  # nothing on-topic enough

        floor = max(self.min_overlap, best_overlap - self.margin)
        candidates = sorted(
            ((-o, r, s) for (o, r, s) in scored if o >= floor),
        )

        chosen: list[str] = []
        seen: set[str] = set()
        for _neg_overlap, _rank, sentence in candidates:
            if sentence in seen:
                continue
            seen.add(sentence)
            chosen.append(sentence)
            if len(chosen) >= self.max_sentences:
                break

        body = " ".join(s if s.endswith((".", "!", "?")) else s + "." for s in chosen)
        return f"Based on the context: {body}"
