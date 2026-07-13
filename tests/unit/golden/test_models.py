import pytest
from pydantic import ValidationError

from rag_flagship.golden.models import GoldenQAPair


def _base(**overrides: object) -> dict:
    payload = {
        "qa_id": "test_001",
        "question": "What is a data subject?",
        "question_language": "en",
        "expected_answer": "A natural person whose data is processed.",
        "expected_source_doc_ids": ["gdpr_en"],
        "expected_locators": ["Article 4 - Definitions"],
        "category": "factual",
        "is_cross_lingual": False,
    }
    payload.update(overrides)
    return payload


def test_valid_factual_pair_is_accepted() -> None:
    pair = GoldenQAPair(**_base())
    assert pair.category == "factual"


def test_out_of_corpus_pair_with_sources_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GoldenQAPair(**_base(category="out_of_corpus"))


def test_out_of_corpus_pair_without_sources_is_accepted() -> None:
    pair = GoldenQAPair(
        **_base(category="out_of_corpus", expected_source_doc_ids=[], expected_locators=[])
    )
    assert pair.expected_source_doc_ids == []


def test_factual_pair_without_sources_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GoldenQAPair(**_base(expected_source_doc_ids=[], expected_locators=[]))
