from dataclasses import dataclass, field

from llama_index.core.schema import NodeWithScore, TextNode

from rag_flagship.reranking.pipeline import PairInput, rerank


@dataclass
class FakeReranker:
    """Scores each pair by how many words it shares with the query, so
    tests can construct predictable rankings without a real model."""

    calls: list[list[PairInput]] = field(default_factory=list)

    def predict(self, inputs: list[PairInput]) -> list[float]:
        self.calls.append(inputs)
        scores = []
        for query, document in inputs:
            query_words = set(query.lower().split())
            doc_words = set(document.lower().split())
            scores.append(float(len(query_words & doc_words)))
        return scores


def _node(text: str, doc_id: str) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text, metadata={"doc_id": doc_id}), score=0.1)


def test_reranks_by_relevance_not_original_order() -> None:
    candidates = [
        _node("cats sleep all day", "off_topic"),
        _node("data subject access request rights", "on_topic"),
    ]

    results = rerank("data subject access rights", candidates, FakeReranker(), top_k=2)

    assert results[0].node.metadata["doc_id"] == "on_topic"
    assert results[1].node.metadata["doc_id"] == "off_topic"


def test_top_k_truncates_results() -> None:
    candidates = [_node(f"passage number {i}", f"doc_{i}") for i in range(5)]

    results = rerank("passage", candidates, FakeReranker(), top_k=2)

    assert len(results) == 2


def test_empty_candidates_returns_empty_without_calling_reranker() -> None:
    reranker = FakeReranker()

    results = rerank("anything", [], reranker, top_k=5)

    assert results == []
    assert reranker.calls == []


def test_scores_are_overwritten_with_cross_encoder_scores() -> None:
    candidates = [_node("data subject access rights", "doc")]

    results = rerank("data subject access", candidates, FakeReranker(), top_k=1)

    assert results[0].score == 3.0


def test_reranker_is_called_with_query_paired_to_each_candidate_text() -> None:
    reranker = FakeReranker()
    candidates = [_node("first passage", "a"), _node("second passage", "b")]

    rerank("my query", candidates, reranker, top_k=2)

    assert reranker.calls == [[("my query", "first passage"), ("my query", "second passage")]]
