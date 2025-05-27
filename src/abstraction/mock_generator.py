"""Baseline generator: return the context sentence with the most query-token
overlap, verbatim. Grounded by construction but brittle, since on an off-topic
query it still hands back whatever sentence was closest."""
from __future__ import annotations

from typing import Sequence

from ..input_layer.schema import RetrievedDoc
from .generator_interface import (
    ABSTAIN, DEFAULT_SEED, Generator, content_tokens, split_sentences)


class ExtractiveGenerator(Generator):
    name = "extractive"

    def generate(self, query: str, contexts: Sequence[RetrievedDoc],
                 *, seed: int = DEFAULT_SEED) -> str:
        if not contexts:
            return ABSTAIN
        q_tokens = set(content_tokens(query))
        best_sentence = ""
        best_overlap = -1
        best_rank = 10**9
        for rank, ctx in enumerate(contexts):
            for sentence in split_sentences(ctx.text):
                overlap = len(q_tokens & set(content_tokens(sentence)))
                # Deterministic tie-break: higher overlap, then earlier context,
                # then lexicographically smaller sentence.
                if (overlap > best_overlap
                        or (overlap == best_overlap and rank < best_rank)
                        or (overlap == best_overlap and rank == best_rank
                            and sentence < best_sentence)):
                    best_overlap = overlap
                    best_rank = rank
                    best_sentence = sentence
        if best_overlap <= 0:
            # nothing overlaps; fall back to the top doc's first sentence.
            # this is the part that hallucinates.
            sentences = split_sentences(contexts[0].text)
            return sentences[0] if sentences else ABSTAIN
        return best_sentence
