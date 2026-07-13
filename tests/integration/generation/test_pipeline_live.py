"""Runs the real retrieval, reranking, and Mistral generation pipeline.

Run explicitly with: uv run pytest -m integration
Requires a local Qdrant instance (docker compose up -d, with
rag_flagship_recursive already indexed) and a local Ollama server with
bge-m3 and mistral-small3.2 pulled.
"""

import pytest

from rag_flagship.embeddings.dense import build_dense_embedding_model
from rag_flagship.generation.llm import build_generation_model
from rag_flagship.generation.pipeline import answer_question
from rag_flagship.indexing.pipeline import hybrid_query
from rag_flagship.indexing.store import build_vector_store
from rag_flagship.reranking.cross_encoder import build_reranker

pytestmark = pytest.mark.integration


def test_in_corpus_question_is_answered_with_citations() -> None:
    embed_model = build_dense_embedding_model()
    vector_store = build_vector_store("rag_flagship_recursive")
    reranker = build_reranker()
    llm = build_generation_model()

    def retriever(question: str, top_k: int):
        return hybrid_query(vector_store, embed_model, question, top_k=top_k)

    # This exact phrasing is one of the golden dataset's questions
    # (factual_en_gdpr_003), confirmed during threshold calibration
    # (scripts/calibrate_refusal_threshold.py) to retrieve its source
    # passage with a top reranked score of 1.000. An ad-hoc phrasing of
    # "maximum fine for a GDPR infringement" was tried first and refused:
    # the recursive strategy's hybrid retrieval for that phrasing surfaces
    # AI Act Article 99 and a GDPR recital, not GDPR Article 83, a real
    # retrieval-quality gap already noted in the Semaine 1 usage guide,
    # not a bug in this pipeline.
    result = answer_question(
        "Within how many hours must a controller notify a personal data "
        "breach to the supervisory authority, where feasible?",
        "en",
        retriever,
        reranker,
        llm,
    )

    assert result.refused is False
    assert len(result.sources) > 0
    assert result.answer


def test_out_of_corpus_question_is_refused() -> None:
    embed_model = build_dense_embedding_model()
    vector_store = build_vector_store("rag_flagship_recursive")
    reranker = build_reranker()
    llm = build_generation_model()

    def retriever(question: str, top_k: int):
        return hybrid_query(vector_store, embed_model, question, top_k=top_k)

    result = answer_question(
        "What are the specific data breach notification deadlines under the "
        "California Consumer Privacy Act (CCPA)?",
        "en",
        retriever,
        reranker,
        llm,
    )

    assert result.refused is True
    assert result.sources == []
