"""Orchestrates retrieval, reranking, the refusal check, and generation.

Retrieval and the LLM are injected as callables/protocols (matching the
pattern already used by `chunking.semantic` and `reranking.pipeline`),
so the branching logic, especially the refusal path, is unit-testable
without a live Qdrant, Ollama, or the real reranker model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, ConfigDict

from rag_flagship.corpus.manifest import Language
from rag_flagship.generation.prompt import REFUSAL_PHRASE, build_messages
from rag_flagship.reranking.pipeline import Reranker, rerank

DEFAULT_RETRIEVAL_TOP_K = 20
DEFAULT_RERANK_TOP_K = 5

REFUSAL_SCORE_THRESHOLD = 0.27
"""bge-reranker-v2-m3 scores via sentence-transformers are sigmoid-bounded
to [0, 1], not raw logits. See ADR-0005: calibrated with
scripts/calibrate_refusal_threshold.py against all 62 golden questions on
rag_flagship_recursive. The 5 clearest out_of_corpus questions scored at
or below 0.256; the lowest-scoring genuinely answerable question (a
cross-lingual one) scored 0.292. 0.27 sits in that narrow gap. One
out_of_corpus question (a plausible-sounding fabricated AI Act claim)
scored 0.801 and is not caught by this threshold; it relies on the LLM's
own refusal instruction as the second layer."""


class LLMClient(Protocol):
    def chat(self, messages: list[ChatMessage]) -> ChatResponse: ...


Retriever = Callable[[str, int], Sequence[NodeWithScore]]


class AnswerSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    locator: str
    score: float


class AnswerResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    sources: list[AnswerSource]
    refused: bool


def _no_sources_result(question_language: Language) -> AnswerResult:
    return AnswerResult(answer=REFUSAL_PHRASE[question_language], sources=[], refused=True)


def answer_question(
    question: str,
    question_language: Language,
    retriever: Retriever,
    reranker: Reranker,
    llm: LLMClient,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    refusal_threshold: float = REFUSAL_SCORE_THRESHOLD,
) -> AnswerResult:
    candidates = retriever(question, retrieval_top_k)
    if not candidates:
        return _no_sources_result(question_language)

    reranked = rerank(question, candidates, reranker, rerank_top_k)
    top_score = reranked[0].score if reranked and reranked[0].score is not None else float("-inf")
    if top_score < refusal_threshold:
        return _no_sources_result(question_language)

    messages = build_messages(question, question_language, reranked)
    response = llm.chat(messages)

    sources = [
        AnswerSource(
            doc_id=node.node.metadata.get("doc_id", "unknown"),
            locator=node.node.metadata.get("locator", "unknown"),
            score=node.score if node.score is not None else 0.0,
        )
        for node in reranked
    ]
    return AnswerResult(
        answer=response.message.content or "",
        sources=sources,
        refused=False,
    )
