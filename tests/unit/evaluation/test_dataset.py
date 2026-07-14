import pytest
from pydantic import ValidationError

from rag_flagship.evaluation.dataset import build_eval_sample
from rag_flagship.golden.models import GoldenQAPair

GOLDEN = GoldenQAPair(
    qa_id="factual_en_gdpr_001",
    question="What are the conditions for consent under GDPR?",
    question_language="en",
    expected_answer="Consent must be freely given, specific, informed, and unambiguous.",
    expected_source_doc_ids=["gdpr_en"],
    expected_locators=["Article 7"],
    category="factual",
    is_cross_lingual=False,
)


def test_build_eval_sample_carries_qa_id_and_question() -> None:
    sample = build_eval_sample(GOLDEN, answer="An answer.", retrieved_contexts=["context 1"])

    assert sample.qa_id == "factual_en_gdpr_001"
    assert sample.question == GOLDEN.question


def test_build_eval_sample_uses_expected_answer_as_reference() -> None:
    sample = build_eval_sample(GOLDEN, answer="An answer.", retrieved_contexts=["context 1"])

    assert sample.reference == GOLDEN.expected_answer


def test_build_eval_sample_preserves_generated_answer_and_contexts() -> None:
    sample = build_eval_sample(
        GOLDEN, answer="Consent must be unambiguous.", retrieved_contexts=["ctx a", "ctx b"]
    )

    assert sample.answer == "Consent must be unambiguous."
    assert sample.retrieved_contexts == ["ctx a", "ctx b"]


def test_build_eval_sample_is_frozen() -> None:
    sample = build_eval_sample(GOLDEN, answer="An answer.", retrieved_contexts=["context 1"])

    with pytest.raises(ValidationError):
        sample.answer = "mutated"  # type: ignore[misc]
