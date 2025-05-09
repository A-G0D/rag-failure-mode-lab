"""Field checks on documents and queries; raises ValidationError on bad input."""
from __future__ import annotations

from typing import Any

from .schema import Document, Query
from .stress_taxonomy import is_valid_stress_type

MAX_QUERY_LENGTH = 2_000


class ValidationError(ValueError):
    pass


def validate_document(doc: Document) -> Document:
    if not doc.doc_id:
        raise ValidationError("doc_id must be a non-empty string")
    if not doc.text.strip():
        raise ValidationError(f"{doc.doc_id}: text must be a non-empty string")
    return doc


def validate_query(query: Query) -> Query:
    if not query.query_id:
        raise ValidationError("query_id must be a non-empty string")
    if not query.text.strip():
        raise ValidationError(f"{query.query_id}: text must be a non-empty string")
    if len(query.text) > MAX_QUERY_LENGTH:
        raise ValidationError(f"{query.query_id}: text exceeds {MAX_QUERY_LENGTH} chars")
    if not is_valid_stress_type(query.stress_type.value):
        raise ValidationError(f"{query.query_id}: invalid stress_type {query.stress_type}")
    return query


def validate_raw_document(d: dict[str, Any]) -> Document:
    return validate_document(Document.from_dict(d))


def validate_raw_query(d: dict[str, Any]) -> Query:
    return validate_query(Query.from_dict(d))
