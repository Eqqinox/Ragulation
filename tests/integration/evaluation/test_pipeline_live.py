"""Runs the real RAGAS metrics against the real local Ollama server.

Run explicitly with: uv run pytest -m integration
Requires a local Ollama server with mistral-small3.2 and bge-m3 pulled.
"""

import pytest

from rag_flagship.evaluation.dataset import EvalSample
from rag_flagship.evaluation.judge import (
    build_judge_client,
    build_judge_embeddings,
    build_judge_llm,
)
from rag_flagship.evaluation.pipeline import run_ragas_eval

pytestmark = pytest.mark.integration


def test_faithful_answer_scores_highly_on_all_four_metrics() -> None:
    client = build_judge_client()
    llm = build_judge_llm(client)
    embeddings = build_judge_embeddings()

    sample = EvalSample(
        qa_id="factual_en_gdpr_003",
        question="What are the conditions for consent under GDPR?",
        answer="Consent must be freely given, specific, informed, and unambiguous.",
        retrieved_contexts=[
            "Consent should be given by a clear affirmative act establishing a "
            "freely given, specific, informed and unambiguous indication of the "
            "data subject agreement."
        ],
        reference="Consent must be freely given, specific, informed, and unambiguous.",
    )

    results = run_ragas_eval([sample], llm, embeddings)

    assert len(results) == 1
    result = results[0]
    assert result.qa_id == "factual_en_gdpr_003"
    assert result.faithfulness >= 0.8
    assert result.answer_relevancy >= 0.5
    assert result.context_precision >= 0.8
    assert result.context_recall >= 0.8


def test_unfaithful_answer_scores_low_on_faithfulness() -> None:
    client = build_judge_client()
    llm = build_judge_llm(client)
    embeddings = build_judge_embeddings()

    sample = EvalSample(
        qa_id="fabricated_001",
        question="What are the conditions for consent under GDPR?",
        answer="Consent is valid as long as it is given verbally to any employee.",
        retrieved_contexts=[
            "Consent should be given by a clear affirmative act establishing a "
            "freely given, specific, informed and unambiguous indication of the "
            "data subject agreement."
        ],
        reference="Consent must be freely given, specific, informed, and unambiguous.",
    )

    results = run_ragas_eval([sample], llm, embeddings)

    assert results[0].faithfulness <= 0.5
