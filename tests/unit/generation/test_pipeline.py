from dataclasses import dataclass

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode

from rag_flagship.generation.pipeline import answer_question


def _node(text: str, doc_id: str, locator: str) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(text=text, metadata={"doc_id": doc_id, "locator": locator}),
        score=0.5,
    )


@dataclass
class FakeReranker:
    scores: list[float]
    calls: int = 0

    def predict(self, inputs: list[tuple[str, str]]) -> list[float]:
        self.calls += 1
        return self.scores[: len(inputs)]


@dataclass
class FakeLLM:
    content: str = "The answer is X [gdpr_en, Article 5]."
    calls: int = 0

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        self.calls += 1
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=self.content))


def _retriever(nodes: list[NodeWithScore]):
    def retrieve(question: str, top_k: int) -> list[NodeWithScore]:
        return nodes[:top_k]

    return retrieve


def test_empty_retrieval_refuses_without_calling_reranker_or_llm() -> None:
    reranker = FakeReranker(scores=[])
    llm = FakeLLM()

    result = answer_question("What is the fine for X?", "en", _retriever([]), reranker, llm)

    assert result.refused is True
    assert result.sources == []
    assert reranker.calls == 0
    assert llm.calls == 0


def test_low_score_refuses_without_calling_llm() -> None:
    candidates = [_node("unrelated text", "doc_a", "loc_a")]
    reranker = FakeReranker(scores=[0.01])
    llm = FakeLLM()

    result = answer_question("What is the fine for X?", "en", _retriever(candidates), reranker, llm)

    assert result.refused is True
    assert result.sources == []
    assert llm.calls == 0


def test_high_score_calls_llm_and_returns_structured_sources() -> None:
    candidates = [_node("relevant text", "gdpr_en", "Article 5")]
    reranker = FakeReranker(scores=[0.95])
    llm = FakeLLM(content="Cited answer.")

    result = answer_question(
        "What does Article 5 say?", "en", _retriever(candidates), reranker, llm
    )

    assert result.refused is False
    assert result.answer == "Cited answer."
    assert llm.calls == 1
    assert len(result.sources) == 1
    assert result.sources[0].doc_id == "gdpr_en"
    assert result.sources[0].locator == "Article 5"
    assert result.sources[0].score == 0.95


def test_rerank_top_k_limits_number_of_sources() -> None:
    candidates = [_node(f"text {i}", f"doc_{i}", f"loc_{i}") for i in range(5)]
    reranker = FakeReranker(scores=[0.9, 0.8, 0.7, 0.6, 0.5])
    llm = FakeLLM()

    result = answer_question(
        "Question?", "en", _retriever(candidates), reranker, llm, rerank_top_k=2
    )

    assert len(result.sources) == 2


def test_custom_threshold_is_respected() -> None:
    candidates = [_node("borderline text", "doc_a", "loc_a")]
    reranker = FakeReranker(scores=[0.3])
    llm = FakeLLM()

    result_with_lenient_threshold = answer_question(
        "Q?", "en", _retriever(candidates), reranker, llm, refusal_threshold=0.2
    )
    result_with_strict_threshold = answer_question(
        "Q?", "en", _retriever(candidates), reranker, llm, refusal_threshold=0.5
    )

    assert result_with_lenient_threshold.refused is False
    assert result_with_strict_threshold.refused is True
