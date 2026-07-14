"""Scores EvalSamples against the four RAGAS metrics named in
projet_1_rag_flagship.md's Semaine 3 scope: faithfulness, answer
relevancy, context precision, context recall.

The judge LLM and embeddings are injected (built by
rag_flagship.evaluation.judge), not constructed here, so this module can
be tested against fakes without a live Ollama server for its own logic;
the metrics themselves are only exercised in integration tests, the same
convention already used for indexing.hybrid_query/dense_query.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from rag_flagship.evaluation.dataset import EvalSample


class EvalResult(BaseModel):
    """One sample's scores across all four metrics, in [0, 1]."""

    model_config = ConfigDict(frozen=True)

    qa_id: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


async def _score_sample(
    sample: EvalSample,
    faithfulness: Faithfulness,
    answer_relevancy: AnswerRelevancy,
    context_precision: ContextPrecision,
    context_recall: ContextRecall,
) -> EvalResult:
    # Scored sequentially, not via asyncio.gather: measured directly
    # against the real local Ollama server (one sample, 4 metrics),
    # concurrent took 106.7s versus 101.0s sequential -- concurrency
    # bought no real speedup here, since a single local Ollama instance
    # serializes GPU inference across requests regardless of how many
    # are in flight; sequential is simpler with no downside.
    faithfulness_result = await faithfulness.ascore(
        user_input=sample.question,
        response=sample.answer,
        retrieved_contexts=sample.retrieved_contexts,
    )
    relevancy_result = await answer_relevancy.ascore(
        user_input=sample.question, response=sample.answer
    )
    precision_result = await context_precision.ascore(
        user_input=sample.question,
        reference=sample.reference,
        retrieved_contexts=sample.retrieved_contexts,
    )
    recall_result = await context_recall.ascore(
        user_input=sample.question,
        retrieved_contexts=sample.retrieved_contexts,
        reference=sample.reference,
    )
    return EvalResult(
        qa_id=sample.qa_id,
        faithfulness=faithfulness_result.value,
        answer_relevancy=relevancy_result.value,
        context_precision=precision_result.value,
        context_recall=recall_result.value,
    )


async def _run_ragas_eval_async(
    samples: Sequence[EvalSample],
    llm: InstructorBaseRagasLLM,
    embeddings: BaseRagasEmbedding,
) -> list[EvalResult]:
    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=llm)
    context_recall = ContextRecall(llm=llm)

    results = []
    for sample in samples:
        results.append(
            await _score_sample(
                sample, faithfulness, answer_relevancy, context_precision, context_recall
            )
        )
    return results


def run_ragas_eval(
    samples: Sequence[EvalSample],
    llm: InstructorBaseRagasLLM,
    embeddings: BaseRagasEmbedding,
) -> list[EvalResult]:
    """Score every sample. Both the four metrics within one sample and
    the samples themselves run sequentially: a single local Ollama
    server serializes GPU inference regardless of concurrent requests,
    so there is nothing to gain from concurrency here (measured
    directly, see _score_sample)."""
    return asyncio.run(_run_ragas_eval_async(samples, llm, embeddings))
