"""Builds the RAGAS judge LLM and embeddings client.

RAGAS's current, non-deprecated metric API (ragas.metrics.collections)
expects an InstructorBaseRagasLLM, built via ragas.llms.llm_factory, and
a BaseRagasEmbedding, built via ragas.embeddings.OpenAIEmbeddings. Both
factories take an OpenAI-compatible client; Ollama serves one directly
at /v1, so no separate LLM/embedding client library is needed beyond
`openai`, which RAGAS itself already depends on. See ADR-0006 for why
this replaced an initially-planned LangChain-based wrapper (itself
deprecated by RAGAS, and incompatible with the current metrics API).

The judge LLM and judge embeddings deliberately use two *different*
clients, not one shared client: AnswerRelevancy's embeddings
(judge_embedding_model_name, bge-m3) always come from the same local
Ollama server used for retrieval (rag_flagship.embeddings), never from
wherever judge_base_url points. This matters in CI, where judge_base_url
points at Ollama Cloud (for structured-output reliability -- see
ADR-0006's addenda) but Ollama Cloud's catalog has no embedding models
at all: a real CI run confirmed this the hard way, 404-ing on
/v1/embeddings, before this split existed.
"""

from __future__ import annotations

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM

from rag_flagship.embeddings.settings import EmbeddingSettings
from rag_flagship.evaluation.settings import EvaluationSettings


def build_judge_client(settings: EvaluationSettings | None = None) -> AsyncOpenAI:
    settings = settings or EvaluationSettings()
    # A real key routes to Ollama Cloud; "ollama" is a dummy value a
    # local Ollama server ignores (no auth required).
    return AsyncOpenAI(
        base_url=f"{settings.judge_base_url}/v1", api_key=settings.judge_api_key or "ollama"
    )


def build_judge_llm(
    client: AsyncOpenAI, settings: EvaluationSettings | None = None
) -> InstructorBaseRagasLLM:
    settings = settings or EvaluationSettings()
    return llm_factory(
        settings.judge_model_name, client=client, max_tokens=settings.judge_max_tokens
    )


def build_judge_embeddings(
    settings: EvaluationSettings | None = None,
    embedding_settings: EmbeddingSettings | None = None,
) -> BaseRagasEmbedding:
    settings = settings or EvaluationSettings()
    embedding_settings = embedding_settings or EmbeddingSettings()
    client = AsyncOpenAI(
        base_url=f"{embedding_settings.base_url}/v1",
        api_key=embedding_settings.api_key or "ollama",
    )
    return OpenAIEmbeddings(client=client, model=settings.judge_embedding_model_name)
