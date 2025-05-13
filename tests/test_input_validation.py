"""Input layer: schema + validation tests."""
import pytest

from src.input_layer.schema import Document, Query, RetrievedDoc
from src.input_layer.stress_taxonomy import StressType, is_valid_stress_type
from src.input_layer.validation import (
    MAX_QUERY_LENGTH, ValidationError, validate_document, validate_query,
)


def test_document_roundtrip():
    d = Document("D1", "some text", "src/x.md", ("tag",))
    assert Document.from_dict(d.as_dict()) == d


def test_query_roundtrip_and_enum():
    q = Query("Q1", "what?", StressType.ADVERSARIAL, "gold", ("D1",))
    back = Query.from_dict(q.as_dict())
    assert back == q
    assert back.stress_type is StressType.ADVERSARIAL


def test_validate_document_rejects_empty_text():
    with pytest.raises(ValidationError):
        validate_document(Document("D1", "   "))


def test_validate_document_rejects_empty_id():
    with pytest.raises(ValidationError):
        validate_document(Document("", "text"))


def test_validate_query_rejects_overlong():
    with pytest.raises(ValidationError):
        validate_query(Query("Q1", "x" * (MAX_QUERY_LENGTH + 1)))


def test_stress_type_enum():
    assert is_valid_stress_type("adversarial")
    assert not is_valid_stress_type("nonsense")
    with pytest.raises(ValueError):
        StressType.from_str("nonsense")


def test_retrieved_doc_as_dict_rounds_score():
    rd = RetrievedDoc("D1", 0.1234567, "t", "s")
    assert rd.as_dict()["score"] == round(0.1234567, 6)
