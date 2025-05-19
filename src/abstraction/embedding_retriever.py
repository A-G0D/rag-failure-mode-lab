"""TF-IDF + cosine retriever (sklearn). Cosine here is just a dot product
since the vectors come out L2-normalized."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shared.determinism import DEFAULT_SEED

from .retriever_interface import Retriever, tokenize


class EmbeddingRetriever(Retriever):
    name = "embedding"

    def __init__(self, *, seed: int = DEFAULT_SEED) -> None:
        super().__init__(seed=seed)
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._row_ids: list[str] = []

    def _build(self) -> None:
        self._row_ids = [d.doc_id for d in self._docs]
        corpus = [" ".join(tokenize(d.text)) for d in self._docs]
        # pre-tokenized input, so str.split is enough and the vocab order is fixed
        self._vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            norm="l2",
            sublinear_tf=True,
        )
        if corpus:
            self._matrix = self._vectorizer.fit_transform(corpus)
        else:  # pragma: no cover - empty corpus guard
            self._matrix = None

    def _score(self, query: str) -> dict[str, float]:
        scores: dict[str, float] = {did: 0.0 for did in self._row_ids}
        if self._vectorizer is None or self._matrix is None:
            return scores
        q = " ".join(tokenize(query))
        if not q:
            return scores
        q_vec = self._vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        for did, sim in zip(self._row_ids, sims):
            scores[did] = float(sim)
        return scores
