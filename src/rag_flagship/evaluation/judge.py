"""Builds the RAGAS judge LLM and embeddings client.

RAGAS's current, non-deprecated metric API (ragas.metrics.collections)
expects an InstructorBaseRagasLLM, built via ragas.llms.llm_factory, and
a BaseRagasEmbedding, built via ragas.embeddings.OpenAIEmbeddings. Both
factories take an OpenAI-compatible client; Ollama serves one directly
at /v1, so no separate LLM/embedding client library is needed beyond
`openai`, which RAGAS itself already depends on. See ADR-0006 for why
this replaced an initially-planned LangChain-based wrapper (itself
deprecated by RAGAS, and incompatible with the current metrics API).
"""

from __future__ import annotations

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.llms.base import InstructorBaseRagasLLM

from rag_flagship.evaluation.settings import EvaluationSettings


def build_judge_client(settings: EvaluationSettings | None = None) -> AsyncOpenAI:
    settings = settings or EvaluationSettings()
    return AsyncOpenAI(base_url=f"{settings.base_url}/v1", api_key="ollama")


def build_judge_llm(
    client: AsyncOpenAI, settings: EvaluationSettings | None = None
) -> InstructorBaseRagasLLM:
    settings = settings or EvaluationSettings()
    return llm_factory(
        settings.judge_model_name, client=client, max_tokens=settings.judge_max_tokens
    )


def build_judge_embeddings(
    client: AsyncOpenAI, settings: EvaluationSettings | None = None
) -> BaseRagasEmbedding:
    settings = settings or EvaluationSettings()
    return OpenAIEmbeddings(client=client, model=settings.judge_embedding_model_name)
