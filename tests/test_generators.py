"""Generator tests: extractive baseline + grounding-aware template + contract."""
import pytest

from src.abstraction.generator_interface import ABSTAIN, Generator, split_sentences
from src.abstraction.mock_generator import ExtractiveGenerator
from src.abstraction.template_generator import TemplateGenerator
from src.input_layer.schema import RetrievedDoc

CTX = [
    RetrievedDoc("D1", 1.0,
                 "A credit check compares the balance against the limit. "
                 "Orders under the limit are approved automatically.", "s"),
    RetrievedDoc("D2", 0.5, "Goods receipt increases on-hand inventory.", "s"),
]


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        Generator()  # type: ignore[abstract]


def test_extractive_returns_context_sentence():
    ans = ExtractiveGenerator().generate("credit check limit", CTX)
    # The answer must be a verbatim sentence from some context.
    all_sentences = [s for c in CTX for s in split_sentences(c.text)]
    assert ans in all_sentences


def test_extractive_abstains_on_empty_context():
    assert ExtractiveGenerator().generate("anything", []) == ABSTAIN


def test_template_abstains_when_off_topic():
    # A query sharing no content tokens with the context -> abstain.
    ans = TemplateGenerator(min_overlap=2).generate("xylophone zebra", CTX)
    assert ans == ABSTAIN


def test_template_answers_when_on_topic():
    ans = TemplateGenerator().generate("credit check limit approved", CTX)
    assert ans != ABSTAIN
    assert ans.startswith("Based on the context:")


def test_generators_repeat():
    for gen in (ExtractiveGenerator(), TemplateGenerator()):
        out = gen.generate("credit check limit", CTX)
        assert gen.generate("credit check limit", CTX) == out
