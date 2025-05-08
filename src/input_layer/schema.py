"""Document / Query / RetrievedDoc dataclasses. Constraints live in
validation.py."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .stress_taxonomy import StressType


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str = ""
    domain_tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source": self.source,
            "domain_tags": list(self.domain_tags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        return cls(
            doc_id=str(d["doc_id"]),
            text=str(d["text"]),
            source=str(d.get("source", "")),
            domain_tags=tuple(d.get("domain_tags", ()) or ()),
        )


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    stress_type: StressType = StressType.NORMAL
    gold_answer: str = ""
    expected_context_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "stress_type": self.stress_type.value,
            "gold_answer": self.gold_answer,
            "expected_context_ids": list(self.expected_context_ids),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Query":
        return cls(
            query_id=str(d["query_id"]),
            text=str(d["text"]),
            stress_type=StressType.from_str(str(d.get("stress_type", "normal"))),
            gold_answer=str(d.get("gold_answer", "")),
            expected_context_ids=tuple(d.get("expected_context_ids", ()) or ()),
        )


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float
    text: str
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(float(self.score), 6),
            "text": self.text,
            "source": self.source,
        }
